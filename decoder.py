"""
Decoder (Receiver): Приём, распаковка и отображение видео
"""

import cv2
import numpy as np
import serial
import time

from config import (
    VIDEO_WIDTH, VIDEO_HEIGHT,
    SERIAL_PORT_RX, BAUD_RATE,
    PACKET_SIZE, FRAME_TIMEOUT
)
from codec import EdgeCodec
from protocol import PacketParser


class Receiver:
    """Главный класс приёмника"""
    
    def __init__(self):
        self.codec = EdgeCodec()
        self.parser = PacketParser()
        self.serial_port = None
        self.running = False
        self.last_frame = None
    
    def open_serial(self) -> bool:
        """Открывает последовательный порт"""
        try:
            self.serial_port = serial.Serial(
                port=SERIAL_PORT_RX,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.01,  # Небольшой таймаут для неблокирующего чтения
            )
            return True
        except serial.SerialException as e:
            print(f"[Receiver] Serial error: {e}")
            return False
    
    def run(self):
        """Главный цикл приёмника"""
        print("[Receiver] Starting...")
        
        # Открываем порт
        if not self.open_serial():
            print("[Receiver] Failed to open serial port!")
            return
        
        print("[Receiver] Serial port opened")
        
        # Создаём пустой кадр для начала
        self.last_frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
        
        self.running = True
        
        stats_time = time.monotonic()
        stats_frames = 0
        stats_packets = 0
        stats_bytes = 0
        
        try:
            while self.running:
                # === Чтение из порта ===
                try:
                    # Читаем доступные данные
                    available = self.serial_port.in_waiting
                    if available > 0:
                        data = self.serial_port.read(min(available, PACKET_SIZE * 4))
                        stats_bytes += len(data)
                        
                        # Добавляем в парсер
                        self.parser.feed(data)
                except serial.SerialException as e:
                    print(f"[Receiver] Read error: {e}")
                    time.sleep(0.1)
                    continue
                
                # === Парсинг пакетов ===
                packets = self.parser.parse_packets()
                
                for frame_id, frame_type, packet_seq, payload in packets:
                    stats_packets += 1
                    
                    # Обрабатываем пакет
                    result = self.parser.process_packet(
                        frame_id, frame_type, packet_seq, payload
                    )
                    
                    if result is not None:
                        # Кадр собран!
                        ftype, compressed_data = result
                        
                        try:
                            # Декодируем
                            frame = self.codec.decode(ftype, compressed_data)
                            self.last_frame = frame
                            stats_frames += 1
                            
                        except Exception as e:
                            print(f"[Receiver] Decode error: {e}")
                
                # === Отображение ===
                if self.last_frame is not None:
                    display = cv2.resize(self.last_frame, (512, 256),
                                        interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Receiver View", display)
                
                # === UI Events ===
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                
                # === Статистика ===
                current_time = time.monotonic()
                if current_time - stats_time >= 5.0:
                    elapsed = current_time - stats_time
                    fps = stats_frames / elapsed
                    pps = stats_packets / elapsed
                    bps = stats_bytes / elapsed
                    print(f"[Receiver] FPS: {fps:.1f}, Packets/s: {pps:.1f}, Bytes/s: {bps:.1f}")
                    stats_time = current_time
                    stats_frames = 0
                    stats_packets = 0
                    stats_bytes = 0
        
        except KeyboardInterrupt:
            print("\n[Receiver] Interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.running = False
        
        if self.serial_port is not None:
            self.serial_port.close()
        
        cv2.destroyAllWindows()
        print("[Receiver] Stopped")


def main():
    """Точка входа для decoder"""
    receiver = Receiver()
    receiver.run()


if __name__ == "__main__":
    main()