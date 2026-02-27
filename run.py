#!/usr/bin/env python3
"""
Оркестратор для запуска Encoder и Decoder как отдельных процессов
"""

import sys
import signal
import time
from multiprocessing import Process, Event

# Глобальный флаг для graceful shutdown
shutdown_event = Event()


def run_encoder():
    """Запуск encoder в отдельном процессе"""
    from encoder import Sender
    
    sender = Sender()
    
    # Проверяем shutdown event периодически
    original_run = sender.run
    
    def wrapped_run():
        try:
            original_run()
        except Exception as e:
            print(f"[Encoder Process] Error: {e}")
    
    wrapped_run()


def run_decoder():
    """Запуск decoder в отдельном процессе"""
    from decoder import Receiver
    
    receiver = Receiver()
    
    try:
        receiver.run()
    except Exception as e:
        print(f"[Decoder Process] Error: {e}")


def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown"""
    print("\n[Main] Shutdown signal received...")
    shutdown_event.set()


def main():
    """Главная функция оркестратора"""
    print("=" * 60)
    print("Edge Video Streaming System")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Resolution: 128x64")
    print(f"  Target packets/sec: ~25")
    print(f"  Packet size: 128 bytes")
    print(f"  Baud rate: 38400")
    print()
    print("Press 'q' or ESC in any video window to quit")
    print("Press Ctrl+C in terminal to force quit")
    print()
    print("=" * 60)
    
    # Устанавливаем обработчики сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Создаём процессы
    encoder_process = Process(target=run_encoder, name="Encoder")
    decoder_process = Process(target=run_decoder, name="Decoder")
    
    # Запускаем
    print("[Main] Starting Encoder process...")
    encoder_process.start()
    
    time.sleep(0.5)  # Небольшая задержка между запусками
    
    print("[Main] Starting Decoder process...")
    decoder_process.start()
    
    print("[Main] Both processes running")
    print()
    
    # Ждём завершения
    try:
        while encoder_process.is_alive() or decoder_process.is_alive():
            # Проверяем shutdown
            if shutdown_event.is_set():
                break
            
            # Проверяем, не завершился ли какой-то процесс
            if not encoder_process.is_alive() and decoder_process.is_alive():
                print("[Main] Encoder stopped, stopping Decoder...")
                decoder_process.terminate()
                break
            
            if not decoder_process.is_alive() and encoder_process.is_alive():
                print("[Main] Decoder stopped, stopping Encoder...")
                encoder_process.terminate()
                break
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt")
    
    # Завершаем процессы
    print("[Main] Stopping processes...")
    
    if encoder_process.is_alive():
        encoder_process.terminate()
        encoder_process.join(timeout=2)
        if encoder_process.is_alive():
            encoder_process.kill()
    
    if decoder_process.is_alive():
        decoder_process.terminate()
        decoder_process.join(timeout=2)
        if decoder_process.is_alive():
            decoder_process.kill()
    
    print("[Main] All processes stopped")
    print("[Main] Goodbye!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())