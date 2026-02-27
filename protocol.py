"""
Протокол упаковки/распаковки пакетов
"""

import struct
from typing import Tuple, Optional, List
from config import (
    SYNC_MAGIC, PACKET_SIZE, HEADER_SIZE, PAYLOAD_SIZE,
    FRAME_TYPE_KEY, FRAME_TYPE_DELTA, FRAME_TYPE_CONTINUATION
)

# Маркер начала кадра
FRAME_START_MARKER = b'\xFE\xFD'


class PacketPacker:
    """
    Упаковщик кадров в пакеты
    """
    
    def __init__(self):
        self.frame_id: int = 0
        self.pending_data: bytearray = bytearray()
        self.current_frame_seq: int = 0
    
    def add_frame(self, data: bytes, frame_type: int):
        """
        Добавляет кадр в очередь
        
        Формат в потоке (little-endian для length):
        [MARKER: 2B][type: 1B][len_lo: 1B][len_hi: 1B][data: NB][checksum: 1B]
        """
        length = len(data)
        
        # Контрольная сумма
        checksum = 0
        for b in data:
            checksum ^= b
        checksum &= 0xFF
        
        # Формируем кадр
        self.pending_data.extend(FRAME_START_MARKER)       # 2 bytes
        self.pending_data.append(frame_type & 0xFF)        # 1 byte
        self.pending_data.append(length & 0xFF)            # len_lo
        self.pending_data.append((length >> 8) & 0xFF)     # len_hi
        self.pending_data.extend(data)                      # N bytes
        self.pending_data.append(checksum)                  # 1 byte
        
        self.frame_id = (self.frame_id + 1) & 0xFF
    
    def get_next_packet(self) -> Optional[bytes]:
        """
        Формирует пакет ровно PACKET_SIZE байт
        """
        if len(self.pending_data) < PAYLOAD_SIZE:
            return None
        
        payload = bytes(self.pending_data[:PAYLOAD_SIZE])
        del self.pending_data[:PAYLOAD_SIZE]
        
        # Собираем пакет
        packet = bytearray(SYNC_MAGIC)
        packet.append(self.frame_id & 0xFF)
        packet.append(FRAME_TYPE_CONTINUATION)
        packet.append(self.current_frame_seq & 0xFF)
        packet.extend(payload)
        
        self.current_frame_seq = (self.current_frame_seq + 1) & 0xFF
        
        return bytes(packet)
    
    def has_data(self) -> bool:
        return len(self.pending_data) >= PAYLOAD_SIZE
    
    def buffer_size(self) -> int:
        return len(self.pending_data)
    
    def clear(self):
        self.pending_data.clear()
        self.current_frame_seq = 0


class StreamUnpacker:
    """
    Распаковщик потока
    """
    
    def __init__(self):
        self.buffer: bytearray = bytearray()
        self.frames_decoded = 0
        self.frames_dropped = 0
    
    def add_packet(self, packet: bytes) -> List[Tuple[int, bytes]]:
        """
        Добавляет пакет и возвращает найденные кадры
        """
        if len(packet) < PACKET_SIZE:
            return []
        
        if packet[:2] != SYNC_MAGIC:
            return []
        
        # Добавляем payload в буфер
        payload = packet[HEADER_SIZE:PACKET_SIZE]
        self.buffer.extend(payload)
        
        # Извлекаем кадры
        frames = []
        
        safety_counter = 0
        max_iterations = 20
        
        while safety_counter < max_iterations:
            safety_counter += 1
            
            frame = self._try_extract_frame()
            if frame is None:
                break
            frames.append(frame)
            self.frames_decoded += 1
        
        # Защита от переполнения
        max_buffer_size = PAYLOAD_SIZE * 30
        if len(self.buffer) > max_buffer_size:
            last_pos = self._find_last_marker()
            if last_pos > 0:
                self.buffer = self.buffer[last_pos:]
            elif len(self.buffer) > PAYLOAD_SIZE * 10:
                self.buffer = self.buffer[-PAYLOAD_SIZE * 5:]
        
        return frames
    
    def _find_last_marker(self) -> int:
        """Находит позицию последнего маркера"""
        pos = -1
        for i in range(len(self.buffer) - 1):
            if self.buffer[i] == FRAME_START_MARKER[0] and self.buffer[i+1] == FRAME_START_MARKER[1]:
                pos = i
        return pos
    
    def _try_extract_frame(self) -> Optional[Tuple[int, bytes]]:
        """
        Пытается извлечь один кадр
        """
        # Ищем маркер
        marker_pos = -1
        for i in range(len(self.buffer) - 1):
            if self.buffer[i] == FRAME_START_MARKER[0] and self.buffer[i+1] == FRAME_START_MARKER[1]:
                marker_pos = i
                break
        
        if marker_pos < 0:
            return None
        
        # Удаляем мусор до маркера
        if marker_pos > 0:
            del self.buffer[:marker_pos]
        
        # Минимальный размер: marker(2) + type(1) + len(2) + checksum(1) = 6
        if len(self.buffer) < 6:
            return None
        
        # Читаем заголовок (little-endian length)
        frame_type = self.buffer[2]
        length = self.buffer[3] | (self.buffer[4] << 8)  # lo | (hi << 8)
        
        # Валидация типа
        if frame_type not in (FRAME_TYPE_KEY, FRAME_TYPE_DELTA):
            del self.buffer[:1]
            return None
        
        # Защита от неправильной длины
        if length > 2000 or length == 0:
            del self.buffer[:1]
            return None
        
        # Проверяем, есть ли все данные
        total_size = 5 + length + 1  # header(5) + data + checksum(1)
        if len(self.buffer) < total_size:
            return None
        
        # Извлекаем данные
        data = bytes(self.buffer[5:5 + length])
        checksum_received = self.buffer[5 + length]
        
        # Удаляем обработанные данные
        del self.buffer[:total_size]
        
        # Проверяем контрольную сумму
        checksum_calc = 0
        for b in data:
            checksum_calc ^= b
        checksum_calc &= 0xFF
        
        if checksum_received != checksum_calc:
            self.frames_dropped += 1
            return None
        
        return (frame_type, data)
    
    def reset(self):
        self.buffer.clear()
        self.frames_decoded = 0
        self.frames_dropped = 0


class PacketParser:
    """
    Парсер входящего потока байт
    """
    
    def __init__(self):
        self.buffer: bytearray = bytearray()
    
    def add_bytes(self, data: bytes) -> List[bytes]:
        """
        Добавляет байты и возвращает найденные пакеты
        """
        self.buffer.extend(data)
        packets = []
        
        while True:
            sync_pos = -1
            for i in range(len(self.buffer) - 1):
                if self.buffer[i] == SYNC_MAGIC[0] and self.buffer[i+1] == SYNC_MAGIC[1]:
                    sync_pos = i
                    break
            
            if sync_pos < 0:
                if len(self.buffer) > 1:
                    self.buffer = self.buffer[-1:]
                break
            
            if sync_pos > 0:
                del self.buffer[:sync_pos]
            
            if len(self.buffer) < PACKET_SIZE:
                break
            
            packet = bytes(self.buffer[:PACKET_SIZE])
            del self.buffer[:PACKET_SIZE]
            packets.append(packet)
        
        return packets
    
    def reset(self):
        self.buffer.clear()