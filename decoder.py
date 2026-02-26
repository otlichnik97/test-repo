"""
Decoder (Receiver): Прием, распаковка и отображение видео
"""

import cv2
import numpy as np
import serial
import time
from typing import Optional
from config import (
    VIDEO_WIDTH, VIDEO_HEIGHT,
    SERIAL_PORT_RX, BAUD_RATE, SERIAL_TIMEOUT,
    DISPLAY_SCALE, WINDOW_NAME_RECEIVER,
    DEBUG, SHOW_STATS, STATS_INTERVAL
)
from codec import decompress
from protocol import PacketParser, FrameAssembler, Packet


class Decoder:
    """Основной класс декодера"""
    
    def __init__(self):
        self._serial: Optional[serial.Serial] = None
        self._parser = PacketParser()
        self._assembler = FrameAssembler()
        self._running = False
        
        # Последний успешно декодированный кадр
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_id = -1
        
        # Статистика
        self._stats_last_time = time.time()
        self._stats_packets_received = 0
        self._stats_frames_decoded = 0
        self._stats_frames_corrupted = 0
        self._stats_bytes_received = 0
    
    def _open_serial(self) -> bool:
        """Открывает последовательный порт"""
        try:
            self._serial = serial.Serial(
                port=SERIAL_PORT_RX,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=SERIAL_TIMEOUT
            )
            print(f"[DECODER] Serial port {SERIAL_PORT_RX} opened at {BAUD_RATE} baud")
            return True
        except serial.SerialException as e:
            print(f"[DECODER] Failed to open serial port: {e}")
            return False
    
    def _close_serial(self):
        """Закрывает последовательный порт"""
        if self._serial and self._serial.is_open:
            self._serial.close()
            print("[DECODER] Serial port closed")
    
    def _create_placeholder(self) -> np.ndarray:
        """Создает placeholder изображение когда нет данных"""
        img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
        
        # Рисуем текст "NO SIGNAL"
        cv2.putText(
            img, "NO", (40, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128,), 1
        )
        cv2.putText(
            img, "SIGNAL", (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,), 1
        )
        
        return img
    
    def _print_stats(self):
        """Выводит статистику"""
        now = time.time()
        elapsed = now - self._stats_last_time
        
        if elapsed >= STATS_INTERVAL:
            fps = self._stats_frames_decoded / elapsed
            bytes_per_sec = self._stats_bytes_received / elapsed
            corrupt_rate = (
                self._stats_frames_corrupted / 
                max(1, self._stats_frames_decoded + self._stats_frames_corrupted)
            ) * 100
            
            print(f"[DECODER] Decoded: {fps:.1f} fps | "
                  f"Packets: {self._stats_packets_received} | "
                  f"Corrupted: {corrupt_rate:.1f}% | "
                  f"Bandwidth: {bytes_per_sec:.0f} B/s")
            
            self._stats_last_time = now
            self._stats_packets_received = 0
            self._stats_frames_decoded = 0
            self._stats_frames_corrupted = 0
            self._stats_bytes_received = 0
    
    def run(self):
        """Главный цикл декодера"""
        # Открываем последовательный порт
        if not self._open_serial():
            return
        
        self._running = True
        placeholder = self._create_placeholder()
        
        print("[DECODER] Waiting for frames...")
        
        try:
            while self._running:
                # Читаем данные из порта
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    self._stats_bytes_received += len(data)
                    
                    # Парсим пакеты
                    packets = self._parser.feed(data)
                    
                    for packet in packets:
                        self._stats_packets_received += 1
                        
                        if DEBUG:
                            print(f"[DECODER] Packet: frame={packet.frame_id}, "
                                  f"seq={packet.packet_seq}, len={packet.data_len}")
                        
                        # Собираем кадр
                        frame_data = self._assembler.add_packet(packet)
                        
                        if frame_data:
                            # Декодируем
                            decoded = decompress(frame_data)
                            
                            if decoded is not None:
                                self._last_frame = decoded
                                self._last_frame_id = packet.frame_id
                                self._stats_frames_decoded += 1
                                
                                if DEBUG:
                                    print(f"[DECODER] Frame {packet.frame_id} decoded "
                                          f"({len(frame_data)} bytes)")
                            else:
                                self._stats_frames_corrupted += 1
                                if DEBUG:
                                    print(f"[DECODER] Frame {packet.frame_id} corrupted")
                
                # Отображаем
                display_frame = self._last_frame if self._last_frame is not None else placeholder
                
                display = cv2.resize(
                    display_frame,
                    (VIDEO_WIDTH * DISPLAY_SCALE, VIDEO_HEIGHT * DISPLAY_SCALE),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Добавляем информацию
                if self._last_frame is not None:
                    info_text = f"Frame: {self._last_frame_id}"
                    cv2.putText(
                        display, info_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1
                    )
                
                cv2.imshow(WINDOW_NAME_RECEIVER, display)
                
                # Проверяем выход
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                
                # Статистика
                if SHOW_STATS:
                    self._print_stats()
        
        except KeyboardInterrupt:
            print("\n[DECODER] Interrupted by user")
        
        finally:
            self._running = False
            self._close_serial()
            cv2.destroyWindow(WINDOW_NAME_RECEIVER)
            print("[DECODER] Stopped")
    
    def stop(self):
        """Останавливает декодер"""
        self._running = False


def run_decoder():
    """Точка входа для запуска декодера"""
    decoder = Decoder()
    decoder.run()


if __name__ == "__main__":
    run_decoder()