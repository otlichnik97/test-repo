"""
Декодер: прием данных, распаковка и отображение
Реализует логику "Low Latency" - приоритет свежим кадрам
"""

import cv2
import numpy as np
import serial
import threading
import time
from typing import Optional
from collections import deque

from config import (
    SERIAL_PORT_RX, SERIAL_BAUDRATE,
    FRAME_WIDTH, FRAME_HEIGHT,
    PACKET_SIZE, PAYLOAD_SIZE,
    DEBUG_PRINT_STATS, STATS_INTERVAL_SEC,
    DISPLAY_WAIT_MS,
    FRAME_TYPE_KEY, FRAME_TYPE_DELTA
)
from codec import FrameDecoder
from protocol import PacketParser, StreamUnpacker


class SerialReceiver:
    """
    Приемник данных через последовательный порт
    """
    
    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
    
    def open(self) -> bool:
        """Открывает порт"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.01  # Неблокирующее чтение
            )
            print(f"[DECODER] Serial port opened: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[DECODER] Failed to open serial port: {e}")
            return False
    
    def close(self):
        """Закрывает порт"""
        if self.serial:
            self.serial.close()
            self.serial = None
    
    def read(self, size: int = 256) -> bytes:
        """Читает данные из порта"""
        if not self.serial:
            return b''
        try:
            return self.serial.read(size)
        except Exception as e:
            print(f"[DECODER] Serial read error: {e}")
            return b''
    
    def in_waiting(self) -> int:
        """Количество байт в буфере"""
        if not self.serial:
            return 0
        try:
            return self.serial.in_waiting
        except:
            return 0


class DecoderStats:
    """Статистика декодера"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.packets_received = 0
        self.frames_decoded = 0
        self.frames_dropped = 0
        self.bytes_received = 0
        self.decode_errors = 0
        self.start_time = time.time()
    
    def print_stats(self):
        """Выводит статистику"""
        elapsed = time.time() - self.start_time
        if elapsed < 0.1:
            return
        
        pps = self.packets_received / elapsed
        fps = self.frames_decoded / elapsed
        kbps = (self.bytes_received * 8 / 1000) / elapsed
        
        print(f"[STATS] Pkt:{pps:.1f}pps FPS:{fps:.1f} "
              f"{kbps:.1f}kbps Drop:{self.frames_dropped} Err:{self.decode_errors}")


class Decoder:
    """
    Главный класс декодера
    """
    
    def __init__(self):
        self.receiver = SerialReceiver(SERIAL_PORT_RX, SERIAL_BAUDRATE)
        self.packet_parser = PacketParser()
        self.stream_unpacker = StreamUnpacker()
        self.frame_decoder = FrameDecoder()
        self.stats = DecoderStats()
        
        self.running = False
        self.receive_thread: Optional[threading.Thread] = None
        
        # Для отображения
        self.display_frame: Optional[np.ndarray] = None
        self.display_lock = threading.Lock()
        
        # Очередь декодированных кадров (для low latency - только последний)
        self.frame_queue = deque(maxlen=2)
    
    def start(self):
        """Запускает декодер"""
        if not self.receiver.open():
            return False
        
        self.running = True
        self.stats.reset()
        
        # Запуск потока приема
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        
        print("[DECODER] Started")
        return True
    
    def stop(self):
        """Останавливает декодер"""
        self.running = False
        
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        
        self.receiver.close()
        cv2.destroyAllWindows()
        
        print("[DECODER] Stopped")
    
    def _receive_loop(self):
        """
        Поток приема и обработки данных
        """
        while self.running:
            # Читаем данные из порта
            data = self.receiver.read(256)
            
            if not data:
                time.sleep(0.001)
                continue
            
            self.stats.bytes_received += len(data)
            
            # Парсим пакеты
            packets = self.packet_parser.add_bytes(data)
            
            for packet in packets:
                self.stats.packets_received += 1
                
                # Распаковываем кадры из потока
                frames = self.stream_unpacker.add_packet(packet)
                
                for frame_type, frame_data in frames:
                    # Декодируем кадр
                    decoded = self.frame_decoder.decode(frame_data, frame_type)
                    
                    if decoded is not None:
                        self.stats.frames_decoded += 1
                        
                        # Обновляем кадр для отображения
                        with self.display_lock:
                            self.display_frame = decoded.copy()
                    else:
                        self.stats.decode_errors += 1
    
    def run_display(self):
        """
        Основной цикл отображения (в главном потоке)
        """
        last_stats_time = time.time()
        
        # Создаем пустой начальный кадр
        empty_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        
        while self.running:
            # Получаем кадр для отображения
            frame_to_show = None
            with self.display_lock:
                if self.display_frame is not None:
                    frame_to_show = self.display_frame.copy()
            
            if frame_to_show is None:
                frame_to_show = empty_frame
            
            # Увеличиваем для лучшей видимости
            display = cv2.resize(frame_to_show, (512, 256), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Добавляем информацию
            cv2.putText(display, f"Frames: {self.stats.frames_decoded}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            cv2.putText(display, f"Packets: {self.stats.packets_received}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            
            cv2.imshow("Receiver View", display)
            
            # Вывод статистики
            if DEBUG_PRINT_STATS and time.time() - last_stats_time > STATS_INTERVAL_SEC:
                self.stats.print_stats()
                last_stats_time = time.time()
            
            # Обработка нажатий клавиш
            key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
            if key == ord('q') or key == 27:  # q или ESC
                self.running = False
                break


def run_decoder():
    """Функция запуска декодера"""
    decoder = Decoder()
    
    try:
        if decoder.start():
            decoder.run_display()
    except KeyboardInterrupt:
        print("\n[DECODER] Interrupted")
    finally:
        decoder.stop()


if __name__ == "__main__":
    run_decoder()