"""
Протокол упаковки/распаковки пакетов для передачи по последовательному порту
"""

import struct
from typing import List, Optional, Tuple
from dataclasses import dataclass
from config import (
    PACKET_SIZE, SYNC_MAGIC, HEADER_SIZE, PAYLOAD_MAX, DEBUG
)


@dataclass
class Packet:
    """Структура пакета данных"""
    frame_id: int      # ID кадра (0-255, циклический)
    packet_seq: int    # Порядковый номер пакета в кадре
    data_len: int      # Длина полезной нагрузки
    payload: bytes     # Полезная нагрузка
    
    def is_last(self) -> bool:
        """Проверяет, последний ли это пакет в кадре"""
        return self.data_len < PAYLOAD_MAX


class PacketBuilder:
    """Создание пакетов из данных кадра"""
    
    @staticmethod
    def create_packets(frame_data: bytes, frame_id: int) -> List[bytes]:
        """
        Разбивает данные кадра на пакеты.
        
        Args:
            frame_data: Сжатые данные кадра
            frame_id: ID кадра (0-255)
            
        Returns:
            Список сырых пакетов для отправки
        """
        packets = []
        offset = 0
        packet_seq = 0
        
        while offset < len(frame_data):
            # Вычисляем размер payload для этого пакета
            remaining = len(frame_data) - offset
            payload_size = min(remaining, PAYLOAD_MAX)
            
            # Извлекаем payload
            payload = frame_data[offset:offset + payload_size]
            
            # Создаем пакет
            packet = PacketBuilder._build_packet(
                frame_id=frame_id & 0xFF,
                packet_seq=packet_seq & 0xFF,
                payload=payload
            )
            packets.append(packet)
            
            offset += payload_size
            packet_seq += 1
        
        return packets
    
    @staticmethod
    def _build_packet(frame_id: int, packet_seq: int, payload: bytes) -> bytes:
        """
        Собирает один пакет.
        
        Структура:
        - SYNC_MAGIC: 2 байта (0xAA 0xBB)
        - frame_id: 1 байт
        - packet_seq: 1 байт
        - data_len: 1 байт
        - payload: до 123 байт
        
        Args:
            frame_id: ID кадра
            packet_seq: Номер пакета
            payload: Полезная нагрузка
            
        Returns:
            Сырой пакет (до 128 байт)
        """
        data_len = len(payload)
        
        # Заголовок
        header = struct.pack(
            '>2sBBB',  # Big-endian: magic(2), frame_id(1), seq(1), len(1)
            SYNC_MAGIC,
            frame_id,
            packet_seq,
            data_len
        )
        
        # Собираем пакет (без паддинга!)
        packet = header + payload
        
        return packet


class PacketParser:
    """Парсинг входящего потока данных"""
    
    def __init__(self):
        self._buffer = bytearray()
        self._sync_found = False
    
    def feed(self, data: bytes) -> List[Packet]:
        """
        Добавляет данные в буфер и извлекает готовые пакеты.
        
        Args:
            data: Входящие байты
            
        Returns:
            Список распознанных пакетов
        """
        self._buffer.extend(data)
        packets = []
        
        while True:
            packet = self._try_extract_packet()
            if packet is None:
                break
            packets.append(packet)
        
        # Ограничиваем размер буфера (защита от переполнения)
        if len(self._buffer) > PACKET_SIZE * 10:
            # Ищем последний SYNC_MAGIC и отбрасываем все до него
            last_sync = self._buffer.rfind(SYNC_MAGIC)
            if last_sync > 0:
                self._buffer = self._buffer[last_sync:]
            elif last_sync < 0:
                self._buffer.clear()
        
        return packets
    
    def _try_extract_packet(self) -> Optional[Packet]:
        """
        Пытается извлечь один пакет из буфера.
        
        Returns:
            Распознанный пакет или None
        """
        # Ищем SYNC_MAGIC
        while len(self._buffer) >= HEADER_SIZE:
            sync_pos = self._buffer.find(SYNC_MAGIC)
            
            if sync_pos < 0:
                # SYNC не найден - оставляем последний байт (может быть частью SYNC)
                if len(self._buffer) > 1:
                    self._buffer = self._buffer[-1:]
                return None
            
            if sync_pos > 0:
                # Отбрасываем мусор перед SYNC
                if DEBUG:
                    print(f"[PROTO] Discarding {sync_pos} bytes before SYNC")
                self._buffer = self._buffer[sync_pos:]
            
            # Проверяем, достаточно ли данных для заголовка
            if len(self._buffer) < HEADER_SIZE:
                return None
            
            # Парсим заголовок
            try:
                _, frame_id, packet_seq, data_len = struct.unpack(
                    '>2sBBB',
                    bytes(self._buffer[:HEADER_SIZE])
                )
            except struct.error:
                self._buffer = self._buffer[2:]  # Пропускаем неверный SYNC
                continue
            
            # Валидация data_len
            if data_len > PAYLOAD_MAX:
                if DEBUG:
                    print(f"[PROTO] Invalid data_len: {data_len}, skipping")
                self._buffer = self._buffer[2:]
                continue
            
            # Проверяем, есть ли полный пакет
            packet_total_size = HEADER_SIZE + data_len
            if len(self._buffer) < packet_total_size:
                return None
            
            # Извлекаем payload
            payload = bytes(self._buffer[HEADER_SIZE:packet_total_size])
            
            # Удаляем пакет из буфера
            self._buffer = self._buffer[packet_total_size:]
            
            return Packet(
                frame_id=frame_id,
                packet_seq=packet_seq,
                data_len=data_len,
                payload=payload
            )
        
        return None
    
    def reset(self):
        """Сбрасывает буфер парсера"""
        self._buffer.clear()
        self._sync_found = False


class FrameAssembler:
    """Сборка кадров из пакетов"""
    
    def __init__(self):
        self._current_frame_id: Optional[int] = None
        self._packets: dict = {}  # packet_seq -> payload
        self._expected_packets: Optional[int] = None
    
    def add_packet(self, packet: Packet) -> Optional[bytes]:
        """
        Добавляет пакет и пытается собрать кадр.
        
        Логика Low Latency:
        - Если пришел пакет с новым frame_id - сбрасываем буфер
        - Приоритет всегда у самого свежего кадра
        
        Args:
            packet: Входящий пакет
            
        Returns:
            Собранные данные кадра или None
        """
        # Проверяем, новый ли это кадр
        if self._current_frame_id is not None:
            # Вычисляем "расстояние" между frame_id (с учетом циклического переполнения)
            diff = (packet.frame_id - self._current_frame_id) & 0xFF
            
            # Если новый кадр (diff > 0 и < 128) - сбрасываем старый
            if 0 < diff < 128:
                if DEBUG and len(self._packets) > 0:
                    print(f"[PROTO] Dropping incomplete frame {self._current_frame_id}, "
                          f"switching to {packet.frame_id}")
                self._reset()
        
        # Начинаем сборку нового кадра
        if self._current_frame_id is None:
            self._current_frame_id = packet.frame_id
        
        # Пропускаем пакеты от старых кадров
        if packet.frame_id != self._current_frame_id:
            return None
        
        # Сохраняем пакет
        self._packets[packet.packet_seq] = packet.payload
        
        # Проверяем, последний ли это пакет
        if packet.is_last():
            self._expected_packets = packet.packet_seq + 1
        
        # Пробуем собрать кадр
        return self._try_assemble()
    
    def _try_assemble(self) -> Optional[bytes]:
        """
        Пытается собрать полный кадр.
        
        Returns:
            Данные кадра или None
        """
        if self._expected_packets is None:
            return None
        
        # Проверяем, все ли пакеты получены
        if len(self._packets) < self._expected_packets:
            return None
        
        # Собираем данные в правильном порядке
        try:
            frame_data = bytearray()
            for seq in range(self._expected_packets):
                if seq not in self._packets:
                    if DEBUG:
                        print(f"[PROTO] Missing packet {seq} in frame {self._current_frame_id}")
                    self._reset()
                    return None
                frame_data.extend(self._packets[seq])
            
            # Успешно собрали кадр
            result = bytes(frame_data)
            self._reset()
            return result
            
        except Exception as e:
            if DEBUG:
                print(f"[PROTO] Assembly error: {e}")
            self._reset()
            return None
    
    def _reset(self):
        """Сбрасывает состояние сборщика"""
        self._current_frame_id = None
        self._packets.clear()
        self._expected_packets = None
    
    def force_reset(self):
        """Принудительный сброс (публичный метод)"""
        self._reset()


# Тестирование протокола
if __name__ == "__main__":
    import os
    
    # Тестовые данные (имитация сжатого кадра)
    test_data = os.urandom(200)  # 200 байт - потребует 2 пакета
    
    print(f"Original data size: {len(test_data)} bytes")
    
    # Создаем пакеты
    packets = PacketBuilder.create_packets(test_data, frame_id=42)
    print(f"Created {len(packets)} packets")
    
    for i, pkt in enumerate(packets):
        print(f"  Packet {i}: {len(pkt)} bytes")
    
    # Симулируем прием
    parser = PacketParser()
    assembler = FrameAssembler()
    
    # Передаем пакеты "по сети" (с возможными фрагментами)
    all_data = b''.join(packets)
    
    # Имитируем фрагментированный прием
    chunk_size = 50
    for i in range(0, len(all_data), chunk_size):
        chunk = all_data[i:i + chunk_size]
        
        parsed_packets = parser.feed(chunk)
        for pkt in parsed_packets:
            print(f"Received packet: frame={pkt.frame_id}, seq={pkt.packet_seq}, len={pkt.data_len}")
            
            frame = assembler.add_packet(pkt)
            if frame:
                print(f"Frame assembled! Size: {len(frame)} bytes")
                print(f"Data match: {frame == test_data}")