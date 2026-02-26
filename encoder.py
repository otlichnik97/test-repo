"""
Encoder (Sender): Захват видео, обработка, сжатие и отправка
"""

import cv2
import numpy as np
import serial
import time
from typing import Optional
from config import (
    VIDEO_WIDTH, VIDEO_HEIGHT,
    SERIAL_PORT_TX, BAUD_RATE,
    CANNY_THRESHOLD1, CANNY_THRESHOLD2,
    DILATE_KERNEL_SIZE, DILATE_ITERATIONS,
    DISPLAY_SCALE, WINDOW_NAME_SENDER,
    TARGET_FPS, MAX_COMPRESSED_SIZE, TARGET_COMPRESSED_SIZE,
    DEBUG, SHOW_STATS, STATS_INTERVAL
)
from codec import compress
from protocol import PacketBuilder


class VideoProcessor:
    """Обработка видеокадров: resize, edge detection, binarization"""
    
    def __init__(self):
        self._dilate_kernel = np.ones(
            (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), 
            np.uint8
        )
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Обрабатывает кадр:
        1. Resize до 128x64
        2. Grayscale
        3. Canny edge detection
        4. Dilate для усиления линий
        5. Бинаризация (0/255)
        
        Args:
            frame: Входной кадр (BGR)
            
        Returns:
            Бинарное изображение (128x64, значения 0 или 255)
        """
        # Resize
        resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        
        # Grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Canny edge detection
        edges = cv2.Canny(gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
        
        # Dilate для усиления тонких линий
        dilated = cv2.dilate(
            edges, 
            self._dilate_kernel, 
            iterations=DILATE_ITERATIONS
        )
        
        # Финальная бинаризация (уже бинарное после Canny, но для надежности)
        _, binary = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
        
        return binary


class Encoder:
    """Основной класс энкодера"""
    
    def __init__(self, camera_id: int = 0):
        self._camera_id = camera_id
        self._processor = VideoProcessor()
        self._frame_id = 0
        self._serial: Optional[serial.Serial] = None
        self._running = False
        
        # Статистика
        self._stats_last_time = time.time()
        self._stats_frames_captured = 0
        self._stats_frames_sent = 0
        self._stats_frames_dropped = 0
        self._stats_bytes_sent = 0
    
    def _open_serial(self) -> bool:
        """Открывает последовательный порт"""
        try:
            self._serial = serial.Serial(
                port=SERIAL_PORT_TX,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=1.0
            )
            print(f"[ENCODER] Serial port {SERIAL_PORT_TX} opened at {BAUD_RATE} baud")
            return True
        except serial.SerialException as e:
            print(f"[ENCODER] Failed to open serial port: {e}")
            return False
    
    def _close_serial(self):
        """Закрывает последовательный порт"""
        if self._serial and self._serial.is_open:
            self._serial.close()
            print("[ENCODER] Serial port closed")
    
    def _send_frame(self, compressed_data: bytes) -> bool:
        """
        Отправляет сжатый кадр через последовательный порт.
        
        Args:
            compressed_data: Сжатые данные кадра
            
        Returns:
            True если отправка успешна
        """
        if not self._serial or not self._serial.is_open:
            return False
        
        # Создаем пакеты
        packets = PacketBuilder.create_packets(compressed_data, self._frame_id)
        
        # Отправляем пакеты
        try:
            for packet in packets:
                self._serial.write(packet)
                self._stats_bytes_sent += len(packet)
            
            self._serial.flush()
            return True
            
        except serial.SerialException as e:
            if DEBUG:
                print(f"[ENCODER] Send error: {e}")
            return False
    
    def _print_stats(self):
        """Выводит статистику"""
        now = time.time()
        elapsed = now - self._stats_last_time
        
        if elapsed >= STATS_INTERVAL:
            fps_captured = self._stats_frames_captured / elapsed
            fps_sent = self._stats_frames_sent / elapsed
            bytes_per_sec = self._stats_bytes_sent / elapsed
            drop_rate = (self._stats_frames_dropped / max(1, self._stats_frames_captured)) * 100
            
            print(f"[ENCODER] Captured: {fps_captured:.1f} fps | "
                  f"Sent: {fps_sent:.1f} fps | "
                  f"Dropped: {drop_rate:.1f}% | "
                  f"Bandwidth: {bytes_per_sec:.0f} B/s")
            
            self._stats_last_time = now
            self._stats_frames_captured = 0
            self._stats_frames_sent = 0
            self._stats_frames_dropped = 0
            self._stats_bytes_sent = 0
    
    def run(self):
        """Главный цикл энкодера"""
        # Открываем камеру
        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            print(f"[ENCODER] Failed to open camera {self._camera_id}")
            return
        
        print(f"[ENCODER] Camera {self._camera_id} opened")
        
        # Настройки камеры для низкого разрешения (экономия CPU)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS * 2)
        
        # Открываем последовательный порт
        if not self._open_serial():
            cap.release()
            return
        
        self._running = True
        frame_interval = 1.0 / TARGET_FPS
        last_frame_time = 0
        
        print(f"[ENCODER] Starting capture at {TARGET_FPS} fps target")
        
        try:
            while self._running:
                # Читаем кадр
                ret, frame = cap.read()
                if not ret:
                    print("[ENCODER] Failed to capture frame")
                    time.sleep(0.01)
                    continue
                
                self._stats_frames_captured += 1
                
                # Контроль частоты кадров
                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    # Пропускаем кадр - слишком рано
                    continue
                
                # Обрабатываем кадр
                binary = self._processor.process(frame)
                
                # Сжимаем
                compressed = compress(binary)
                
                if compressed is None:
                    # Кадр слишком большой - пропускаем
                    self._stats_frames_dropped += 1
                    if DEBUG:
                        print("[ENCODER] Frame dropped: compression failed")
                    continue
                
                if len(compressed) > MAX_COMPRESSED_SIZE:
                    # Кадр слишком большой - пропускаем
                    self._stats_frames_dropped += 1
                    if DEBUG:
                        print(f"[ENCODER] Frame dropped: {len(compressed)} > {MAX_COMPRESSED_SIZE}")
                    continue
                
                # Отправляем
                if self._send_frame(compressed):
                    self._stats_frames_sent += 1
                    self._frame_id = (self._frame_id + 1) & 0xFF
                    last_frame_time = current_time
                else:
                    self._stats_frames_dropped += 1
                
                # Отображаем локально
                display = cv2.resize(
                    binary, 
                    (VIDEO_WIDTH * DISPLAY_SCALE, VIDEO_HEIGHT * DISPLAY_SCALE),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Добавляем информацию
                info_text = f"Size: {len(compressed)}B | FID: {self._frame_id}"
                cv2.putText(
                    display, info_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1
                )
                
                cv2.imshow(WINDOW_NAME_SENDER, display)
                
                # Проверяем выход
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                
                # Статистика
                if SHOW_STATS:
                    self._print_stats()
        
        except KeyboardInterrupt:
            print("\n[ENCODER] Interrupted by user")
        
        finally:
            self._running = False
            cap.release()
            self._close_serial()
            cv2.destroyWindow(WINDOW_NAME_SENDER)
            print("[ENCODER] Stopped")
    
    def stop(self):
        """Останавливает энкодер"""
        self._running = False


def run_encoder(camera_id: int = 0):
    """Точка входа для запуска энкодера"""
    encoder = Encoder(camera_id)
    encoder.run()


if __name__ == "__main__":
    run_encoder()