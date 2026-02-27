"""
Энкодер: захват видео, обработка, сжатие и отправка
Реализует паттерн "Leaky Bucket"
"""

import cv2
import numpy as np
import serial
import threading
import time
import queue
from typing import Optional
from collections import deque

from config import (
    SERIAL_PORT_TX, SERIAL_BAUDRATE,
    FRAME_WIDTH, FRAME_HEIGHT,
    CANNY_THRESHOLD_1, CANNY_THRESHOLD_2,
    DILATE_KERNEL_SIZE, DILATE_ITERATIONS,
    PACKET_SIZE, PAYLOAD_SIZE, PACKETS_PER_SECOND, PACKET_INTERVAL_MS,
    BUCKET_MAX_SIZE,
    DEBUG_PRINT_STATS, STATS_INTERVAL_SEC,
    DISPLAY_WAIT_MS
)
from codec import FrameEncoder
from protocol import PacketPacker


class LeakyBucket:
    """
    Потокобезопасный буфер "Дырявое ведро"
    Данные добавляются с переменной скоростью,
    извлекаются со строго фиксированной скоростью
    """
    
    def __init__(self, max_size: int = BUCKET_MAX_SIZE):
        self.buffer: bytearray = bytearray()
        self.lock = threading.Lock()
        self.max_size = max_size
        self.overflow_count = 0
    
    def add(self, data: bytes) -> bool:
        """
        Добавляет данные в буфер
        
        Returns:
            True если добавлено, False если переполнение
        """
        with self.lock:
            if len(self.buffer) + len(data) > self.max_size:
                # Переполнение - отбрасываем старые данные
                overflow = len(self.buffer) + len(data) - self.max_size
                del self.buffer[:overflow]
                self.overflow_count += 1
            
            self.buffer.extend(data)
            return True
    
    def drain(self, size: int) -> bytes:
        """
        Извлекает данные из буфера
        
        Args:
            size: количество байт для извлечения
            
        Returns:
            Извлеченные данные (может быть меньше size)
        """
        with self.lock:
            actual_size = min(size, len(self.buffer))
            data = bytes(self.buffer[:actual_size])
            del self.buffer[:actual_size]
            return data
    
    def size(self) -> int:
        """Текущий размер буфера"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Очистка буфера"""
        with self.lock:
            self.buffer.clear()


class VideoProcessor:
    """
    Обработчик видео: захват, edge detection, бинаризация
    """
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.dilate_kernel = np.ones(
            (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), 
            np.uint8
        )
    
    def open(self) -> bool:
        """Открывает камеру"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[ENCODER] Failed to open camera {self.camera_id}")
            return False
        
        # Настройка камеры для низкой задержки
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        print(f"[ENCODER] Camera opened: {self.camera_id}")
        return True
    
    def close(self):
        """Закрывает камеру"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def capture_and_process(self) -> Optional[np.ndarray]:
        """
        Захватывает кадр и применяет обработку
        
        Returns:
            Бинарное изображение (64x128) с 0/1 или None
        """
        if not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Преобразование в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ресайз до целевого разрешения
        resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT), 
                            interpolation=cv2.INTER_AREA)
        
        # Canny edge detection
        edges = cv2.Canny(resized, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
        
        # Дилатация для усиления границ
        dilated = cv2.dilate(edges, self.dilate_kernel, 
                            iterations=DILATE_ITERATIONS)
        
        # Бинаризация: 0 или 1
        binary = (dilated > 127).astype(np.uint8)
        
        return binary


class SerialSender:
    """
    Отправщик данных через последовательный порт
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
                timeout=0.01,
                write_timeout=0.1
            )
            print(f"[ENCODER] Serial port opened: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[ENCODER] Failed to open serial port: {e}")
            return False
    
    def close(self):
        """Закрывает порт"""
        if self.serial:
            self.serial.close()
            self.serial = None
    
    def send(self, data: bytes) -> bool:
        """Отправляет данные"""
        if not self.serial:
            return False
        try:
            self.serial.write(data)
            self.serial.flush()
            return True
        except Exception as e:
            print(f"[ENCODER] Serial write error: {e}")
            return False


class EncoderStats:
    """Статистика энкодера"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.frames_captured = 0
        self.frames_encoded = 0
        self.packets_sent = 0
        self.bytes_sent = 0
        self.overflows = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
    
    def print_stats(self):
        """Выводит статистику"""
        elapsed = time.time() - self.start_time
        if elapsed < 0.1:
            return
        
        fps_capture = self.frames_captured / elapsed
        fps_encoded = self.frames_encoded / elapsed
        pps = self.packets_sent / elapsed
        kbps = (self.bytes_sent * 8 / 1000) / elapsed
        
        print(f"[STATS] Cap:{fps_capture:.1f}fps Enc:{fps_encoded:.1f}fps "
              f"Pkt:{pps:.1f}pps {kbps:.1f}kbps OVF:{self.overflows}")


class Encoder:
    """
    Главный класс энкодера
    Управляет потоками захвата и отправки
    """
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.processor = VideoProcessor(camera_id)
        self.sender = SerialSender(SERIAL_PORT_TX, SERIAL_BAUDRATE)
        self.frame_encoder = FrameEncoder()
        self.packer = PacketPacker()
        self.bucket = LeakyBucket(BUCKET_MAX_SIZE)
        self.stats = EncoderStats()
        
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.send_thread: Optional[threading.Thread] = None
        
        # Для отображения
        self.display_frame: Optional[np.ndarray] = None
        self.display_lock = threading.Lock()
    
    def start(self):
        """Запускает энкодер"""
        if not self.processor.open():
            return False
        
        if not self.sender.open():
            self.processor.close()
            return False
        
        self.running = True
        self.stats.reset()
        
        # Запуск потоков
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        
        self.capture_thread.start()
        self.send_thread.start()
        
        print("[ENCODER] Started")
        return True
    
    def stop(self):
        """Останавливает энкодер"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.send_thread:
            self.send_thread.join(timeout=1.0)
        
        self.processor.close()
        self.sender.close()
        cv2.destroyAllWindows()
        
        print("[ENCODER] Stopped")
    
    def _capture_loop(self):
        """
        Поток захвата и обработки видео
        Работает максимально быстро
        """
        while self.running:
            # Захват и обработка
            binary_frame = self.processor.capture_and_process()
            if binary_frame is None:
                time.sleep(0.01)
                continue
            
            self.stats.frames_captured += 1
            
            # Кодирование
            encoded_data, frame_type, frame_num = self.frame_encoder.encode(binary_frame)
            self.stats.frames_encoded += 1
            
            # Добавление в packer
            self.packer.add_frame(encoded_data, frame_type)
            
            # Перекачка данных из packer в bucket
            while self.packer.has_data():
                packet = self.packer.get_next_packet()
                if packet:
                    self.bucket.add(packet)
            
            # Обновление кадра для отображения
            with self.display_lock:
                # Преобразуем 0/1 в 0/255 для отображения
                self.display_frame = (binary_frame * 255).astype(np.uint8)
            
            # Небольшая пауза чтобы не перегружать CPU
            time.sleep(0.001)
    
    def _send_loop(self):
        """
        Поток отправки пакетов
        Строго 25 пакетов в секунду
        """
        interval = PACKET_INTERVAL_MS / 1000.0  # 40 мс
        next_send_time = time.time()
        
        while self.running:
            now = time.time()
            
            if now >= next_send_time:
                # Время отправлять пакет
                # Извлекаем данные из bucket
                data = self.bucket.drain(PACKET_SIZE)
                
                if len(data) == PACKET_SIZE:
                    # Отправляем полный пакет
                    if self.sender.send(data):
                        self.stats.packets_sent += 1
                        self.stats.bytes_sent += len(data)
                
                # Планируем следующую отправку
                next_send_time += interval
                
                # Защита от накопления отставания
                if next_send_time < now - interval:
                    next_send_time = now + interval
            
            # Точный сон до следующей отправки
            sleep_time = next_send_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def run_display(self):
        """
        Основной цикл отображения (должен выполняться в главном потоке)
        """
        last_stats_time = time.time()
        
        while self.running:
            # Получаем кадр для отображения
            frame_to_show = None
            with self.display_lock:
                if self.display_frame is not None:
                    frame_to_show = self.display_frame.copy()
            
            if frame_to_show is not None:
                # Увеличиваем для лучшей видимости
                display = cv2.resize(frame_to_show, (512, 256), 
                                    interpolation=cv2.INTER_NEAREST)
                
                # Добавляем информацию
                cv2.putText(display, f"Bucket: {self.bucket.size()} bytes", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                cv2.putText(display, f"Packets: {self.stats.packets_sent}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                
                cv2.imshow("Sender View", display)
            
            # Вывод статистики
            if DEBUG_PRINT_STATS and time.time() - last_stats_time > STATS_INTERVAL_SEC:
                self.stats.overflows = self.bucket.overflow_count
                self.stats.print_stats()
                last_stats_time = time.time()
            
            # Обработка нажатий клавиш
            key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
            if key == ord('q') or key == 27:  # q или ESC
                self.running = False
                break


def run_encoder(camera_id: int = 0):
    """Функция запуска энкодера"""
    encoder = Encoder(camera_id)
    
    try:
        if encoder.start():
            encoder.run_display()
    except KeyboardInterrupt:
        print("\n[ENCODER] Interrupted")
    finally:
        encoder.stop()


if __name__ == "__main__":
    run_encoder()