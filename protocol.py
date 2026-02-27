"""
Протокол упаковки/распаковки пакетов
Обеспечивает равномерную передачу данных
"""

import struct
import time
from collections import deque
from typing import Optional, Tuple, List
from config import (
    SYNC_MAGIC, PACKET_SIZE, HEADER_SIZE, PAYLOAD_SIZE,
    FRAME_TYPE_KEY, FRAME_TYPE_DELTA, FRAME_TYPE_CONTINUATION,
    PACKET_INTERVAL, MAX_PACKETS_PER_FRAME
)


class PacketBuilder:
    """Создаёт пакеты для передачи"""
    
    def __init__(self):
        self.frame_id = 0
        self.packet_queue = deque()  # Очередь готовых пакетов
        self.last_send_time = 0
    
    def add_frame(self, frame_type: int, data: bytes):
        """
        Добавляет кадр в очередь пакетов.
        Разбивает данные на пакеты по PAYLOAD_SIZE байт.
        """
        # Создаём пакеты для кадра
        packets = []
        offset = 0
        packet_seq = 0
        
        while offset < len(data):
            chunk = data[offset:offset + PAYLOAD_SIZE]
            offset += PAYLOAD_SIZE
            
            # Определяем тип для этого пакета
            if packet_seq == 0:
                pkt_type = frame_type  # Первый пакет кадра
            else:
                pkt_type = FRAME_TYPE_CONTINUATION  # Продолжение
            
            packet = self._build_packet(
                self.frame_id, pkt_type, packet_seq, chunk
            )
            packets.append(packet)
            packet_seq += 1
        
        # Добавляем в очередь
        for pkt in packets:
            self.packet_queue.append(pkt)
        
        # Инкрементируем frame_id (циклический 0-255)
        self.frame_id = (self.frame_id + 1) & 0xFF
    
    def _build_packet(self, frame_id: int, frame_type: int, 
                      packet_seq: int, payload: bytes) -> bytes:
        """
        Строит пакет ровно 128 байт.
        Если payload меньше - дополняем данными для эффективности.
        """
        # Заголовок
        header = struct.pack(
            '>2sBBB',  # Big-endian: SYNC(2) + frame_id(1) + type(1) + seq(1)
            SYNC_MAGIC,
            frame_id,
            frame_type,
            packet_seq
        )
        
        # Payload с выравниванием
        if len(payload) < PAYLOAD_SIZE:
            # Дополняем паттерном для синхронизации (не нули!)
            # Используем длину реальных данных в последнем байте payload
            padding_needed = PAYLOAD_SIZE - len(payload) - 1
            padded = payload + bytes([0x55] * padding_needed) + bytes([len(payload)])
        else:
            padded = payload[:PAYLOAD_SIZE]
        
        return header + padded
    
    def get_next_packet(self) -> Optional[bytes]:
        """
        Возвращает следующий пакет с учётом равномерной передачи.
        Неблокирующая операция.
        """
        current_time = time.monotonic()
        
        # Проверяем, прошло ли достаточно времени
        if current_time - self.last_send_time < PACKET_INTERVAL:
            return None
        
        if self.packet_queue:
            self.last_send_time = current_time
            return self.packet_queue.popleft()
        
        return None
    
    def time_until_next(self) -> float:
        """Возвращает время до следующей возможной отправки"""
        elapsed = time.monotonic() - self.last_send_time
        remaining = PACKET_INTERVAL - elapsed
        return max(0, remaining)
    
    def queue_size(self) -> int:
        """Размер очереди пакетов"""
        return len(self.packet_queue)
    
    def clear_old_frames(self, keep_last: int = 2):
        """
        Очищает старые кадры из очереди, оставляя только последние.
        Предотвращает накопление задержки.
        """
        if self.queue_size() > MAX_PACKETS_PER_FRAME * keep_last:
            # Оставляем только последние пакеты
            packets_to_keep = MAX_PACKETS_PER_FRAME * keep_last
            while len(self.packet_queue) > packets_to_keep:
                self.packet_queue.popleft()


class PacketParser:
    """Парсер входящих пакетов"""
    
    def __init__(self):
        self.buffer = bytearray()
        self.frame_buffers = {}  # frame_id -> {packets: {seq: data}, type: int}
        self.current_frame_id = None
        self.last_complete_frame_id = None
    
    def feed(self, data: bytes):
        """Добавляет данные в буфер"""
        self.buffer.extend(data)
    
    def parse_packets(self) -> List[Tuple[int, int, int, bytes]]:
        """
        Парсит все доступные пакеты из буфера.
        Возвращает список: [(frame_id, frame_type, packet_seq, payload), ...]
        """
        packets = []
        
        while len(self.buffer) >= PACKET_SIZE:
            # Ищем SYNC_MAGIC
            sync_pos = self.buffer.find(SYNC_MAGIC)
            
            if sync_pos == -1:
                # Нет синхронизации, оставляем последний байт (может быть частью SYNC)
                if len(self.buffer) > 1:
                    self.buffer = self.buffer[-1:]
                break
            
            if sync_pos > 0:
                # Отбрасываем мусор до синхронизации
                self.buffer = self.buffer[sync_pos:]
            
            if len(self.buffer) < PACKET_SIZE:
                break
            
            # Извлекаем пакет
            packet = bytes(self.buffer[:PACKET_SIZE])
            self.buffer = self.buffer[PACKET_SIZE:]
            
            # Парсим заголовок
            try:
                _, frame_id, frame_type, packet_seq = struct.unpack(
                    '>2sBBB', packet[:HEADER_SIZE]
                )
                payload = packet[HEADER_SIZE:]
                
                # Извлекаем реальную длину из последнего байта для неполных пакетов
                if frame_type == FRAME_TYPE_CONTINUATION or packet_seq > 0:
                    # Для continuation пакетов проверяем маркер длины
                    pass  # Длина обрабатывается при сборке кадра
                
                packets.append((frame_id, frame_type, packet_seq, payload))
                
            except struct.error:
                continue
        
        return packets
    
    def process_packet(self, frame_id: int, frame_type: int, 
                       packet_seq: int, payload: bytes) -> Optional[Tuple[int, bytes]]:
        """
        Обрабатывает пакет и пытается собрать кадр.
        Возвращает (frame_type, data) если кадр собран, иначе None.
        
        Реализует логику Low Latency: новый frame_id сбрасывает старый буфер.
        """
        # Если пришёл новый кадр, сбрасываем предыдущие
        if self.current_frame_id is not None and frame_id != self.current_frame_id:
            # Проверяем, что это действительно новый кадр (учитываем циклический ID)
            id_diff = (frame_id - self.current_frame_id) & 0xFF
            if id_diff > 0 and id_diff < 128:  # Новый кадр (не старый повтор)
                # Сбрасываем старый буфер
                self.frame_buffers.clear()
        
        self.current_frame_id = frame_id
        
        # Инициализируем буфер для кадра
        if frame_id not in self.frame_buffers:
            self.frame_buffers[frame_id] = {
                'packets': {},
                'type': None,
                'expected_end': False
            }
        
        fb = self.frame_buffers[frame_id]
        
        # Сохраняем тип кадра (из первого пакета)
        if frame_type in (FRAME_TYPE_KEY, FRAME_TYPE_DELTA):
            fb['type'] = frame_type
        
        # Сохраняем payload
        fb['packets'][packet_seq] = payload
        
        # Пытаемся собрать кадр
        # Кадр считается полным, если:
        # 1. Есть пакет seq=0
        # 2. Все пакеты последовательны до максимального seq
        if 0 in fb['packets'] and fb['type'] is not None:
            max_seq = max(fb['packets'].keys())
            
            # Проверяем непрерывность
            all_present = all(i in fb['packets'] for i in range(max_seq + 1))
            
            if all_present:
                # Собираем данные
                full_data = bytearray()
                for seq in range(max_seq + 1):
                    pkt_payload = fb['packets'][seq]
                    
                    # Для последнего пакета извлекаем реальную длину
                    if seq == max_seq:
                        real_len = pkt_payload[-1]
                        if real_len > 0 and real_len < PAYLOAD_SIZE:
                            pkt_payload = pkt_payload[:real_len]
                        # Если real_len == PAYLOAD_SIZE, данные полные
                    
                    full_data.extend(pkt_payload)
                
                # Очищаем буфер этого кадра
                del self.frame_buffers[frame_id]
                self.last_complete_frame_id = frame_id
                
                return (fb['type'], bytes(full_data))
        
        return None
    
    def reset(self):
        """Сброс состояния парсера"""
        self.buffer.clear()
        self.frame_buffers.clear()
        self.current_frame_id = None