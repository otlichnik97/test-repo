"""
Encoder (Sender): Захват, обработка, сжатие и отправка видео
"""

import cv2
import numpy as np
import serial
import time
from typing import Optional

from config import (
    VIDEO_WIDTH, VIDEO_HEIGHT,
    CANNY_THRESHOLD_1, CANNY_THRESHOLD_2,
    DILATE_KERNEL_SIZE, DILATE_ITERATIONS,
    KEYFRAME_INTERVAL,
    SERIAL_PORT_TX, BAUD_RATE,
    TARGET_FPS, PACKET_INTERVAL
)
from codec import EdgeCodec
from protocol import PacketBuilder


class VideoEncoder:
    """Захват и обработка видео"""
    
    def __init__(self):
        self.cap = None
        self.dilate_kernel = np.ones(
            (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), 
            np.uint8
        )
    
    def open(self, camera_id: int = 0) -> bool:
        """Открывает камеру"""
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            return False
        
        # Минимизируем буферизацию камеры
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Устанавливаем минимальное разрешение для скорости
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Захватывает кадр с камеры"""
        if self.cap is None:
            return None
        
        # Пропускаем буферизированные кадры
        self.cap.grab()
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Обрабатывает кадр:
        1. Resize до 128x64
        2. Grayscale
        3. Canny edge detection
        4. Dilate
        5. Бинаризация (строго 0/1)
        """
        # Resize
        resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT), 
                            interpolation=cv2.INTER_AREA)
        
        # Grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Canny edge detection
        edges = cv2.Canny(gray, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
        
        # Dilate для утолщения линий
        dilated = cv2.dilate(edges, self.dilate_kernel, 
                            iterations=DILATE_ITERATIONS)
        
        # Бинаризация: строго 0 или 255
        _, binary = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def close(self):
        """Закрывает камеру"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class Sender:
    """Главный класс отправителя"""
    
    def __init__(self):
        self.encoder = VideoEncoder()
        self.codec = EdgeCodec()
        self.packet_builder = PacketBuilder()
        self.serial_port = None
        self.frame_count = 0
        self.running = False
    
    def open_serial(self) -> bool:
        """Открывает последовательный порт"""
        try:
            self.serial_port = serial.Serial(
                port=SERIAL_PORT_TX,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0,  # Неблокирующий режим
                write_timeout=0.1
            )
            return True
        except serial.SerialException as e:
            print(f"[Sender] Serial error: {e}")
            return False
    
    def run(self):
        """Главный цикл отправителя"""
        print("[Sender] Starting...")
        
        # Открываем камеру
        if not self.encoder.open():
            print("[Sender] Failed to open camera!")
            return
        
        # Открываем порт
        if not self.open_serial():
            print("[Sender] Failed to open serial port!")
            self.encoder.close()
            return
        
        print("[Sender] Camera and serial port opened")
        
        self.running = True
        last_frame_time = 0
        frame_interval = 1.0 / TARGET_FPS
        
        stats_time = time.monotonic()
        stats_frames = 0
        stats_packets = 0
        
        try:
            while self.running:
                current_time = time.monotonic()
                
                # === Отправка пакетов (приоритет - равномерность) ===
                packet = self.packet_builder.get_next_packet()
                if packet is not None:
                    try:
                        self.serial_port.write(packet)
                        stats_packets += 1
                    except serial.SerialException as e:
                        print(f"[Sender] Write error: {e}")
                
                # === Захват и обработка кадров ===
                if current_time - last_frame_time >= frame_interval:
                    # Контролируем размер очереди
                    self.packet_builder.clear_old_frames(keep_last=2)
                    
                    frame = self.encoder.capture_frame()
                    if frame is not None:
                        # Обработка
                        binary = self.encoder.process_frame(frame)
                        
                        # Определяем тип кадра
                        force_keyframe = (self.frame_count % KEYFRAME_INTERVAL == 0)
                        
                        # Кодирование
                        frame_type, compressed = self.codec.encode(binary, force_keyframe)
                        
                        # Добавляем в очередь пакетов
                        self.packet_builder.add_frame(frame_type, compressed)
                        
                        self.frame_count += 1
                        stats_frames += 1
                        last_frame_time = current_time
                        
                        # Отображение
                        display = cv2.resize(binary, (512, 256), 
                                           interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("Sender View", display)
                
                # === UI Events ===
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' или ESC
                    break
                
                # === Статистика ===
                if current_time - stats_time >= 5.0:
                    elapsed = current_time - stats_time
                    fps = stats_frames / elapsed
                    pps = stats_packets / elapsed
                    queue_size = self.packet_builder.queue_size()
                    print(f"[Sender] FPS: {fps:.1f}, Packets/s: {pps:.1f}, Queue: {queue_size}")
                    stats_time = current_time
                    stats_frames = 0
                    stats_packets = 0
                
                # === Минимальный sleep для экономии CPU ===
                wait_time = self.packet_builder.time_until_next()
                if wait_time > 0.001:
                    time.sleep(min(wait_time, 0.005))
        
        except KeyboardInterrupt:
            print("\n[Sender] Interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.running = False
        self.encoder.close()
        
        if self.serial_port is not None:
            self.serial_port.close()
        
        cv2.destroyAllWindows()
        print("[Sender] Stopped")


def main():
    """Точка входа для encoder"""
    sender = Sender()
    sender.run()


if __name__ == "__main__":
    main()