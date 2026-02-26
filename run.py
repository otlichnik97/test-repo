#!/usr/bin/env python3
"""
Оркестратор для запуска системы потоковой передачи видео
"""

import sys
import time
import signal
import argparse
from multiprocessing import Process, Event
from typing import Optional


def run_encoder_process(stop_event: Event, camera_id: int):
    """Функция для процесса энкодера"""
    # Импортируем здесь, чтобы избежать проблем с multiprocessing
    import cv2
    from encoder import Encoder
    
    encoder = Encoder(camera_id)
    
    # Запускаем в отдельном потоке проверку stop_event
    import threading
    
    def check_stop():
        while not stop_event.is_set():
            time.sleep(0.1)
        encoder.stop()
    
    checker = threading.Thread(target=check_stop, daemon=True)
    checker.start()
    
    try:
        encoder.run()
    except Exception as e:
        print(f"[ENCODER PROCESS] Error: {e}")
    finally:
        cv2.destroyAllWindows()


def run_decoder_process(stop_event: Event):
    """Функция для процесса декодера"""
    import cv2
    from decoder import Decoder
    
    decoder = Decoder()
    
    import threading
    
    def check_stop():
        while not stop_event.is_set():
            time.sleep(0.1)
        decoder.stop()
    
    checker = threading.Thread(target=check_stop, daemon=True)
    checker.start()
    
    try:
        decoder.run()
    except Exception as e:
        print(f"[DECODER PROCESS] Error: {e}")
    finally:
        cv2.destroyAllWindows()


class StreamingSystem:
    """Оркестратор системы потоковой передачи"""
    
    def __init__(self):
        self._encoder_process: Optional[Process] = None
        self._decoder_process: Optional[Process] = None
        self._stop_event = Event()
    
    def start_encoder(self, camera_id: int = 0):
        """Запускает процесс энкодера"""
        if self._encoder_process and self._encoder_process.is_alive():
            print("[SYSTEM] Encoder already running")
            return
        
        self._encoder_process = Process(
            target=run_encoder_process,
            args=(self._stop_event, camera_id),
            name="VideoEncoder"
        )
        self._encoder_process.start()
        print(f"[SYSTEM] Encoder started (PID: {self._encoder_process.pid})")
    
    def start_decoder(self):
        """Запускает процесс декодера"""
        if self._decoder_process and self._decoder_process.is_alive():
            print("[SYSTEM] Decoder already running")
            return
        
        self._decoder_process = Process(
            target=run_decoder_process,
            args=(self._stop_event,),
            name="VideoDecoder"
        )
        self._decoder_process.start()
        print(f"[SYSTEM] Decoder started (PID: {self._decoder_process.pid})")
    
    def stop(self):
        """Останавливает все процессы"""
        print("[SYSTEM] Stopping...")
        self._stop_event.set()
        
        # Ждем завершения процессов
        if self._encoder_process and self._encoder_process.is_alive():
            self._encoder_process.join(timeout=3)
            if self._encoder_process.is_alive():
                print("[SYSTEM] Force terminating encoder...")
                self._encoder_process.terminate()
        
        if self._decoder_process and self._decoder_process.is_alive():
            self._decoder_process.join(timeout=3)
            if self._decoder_process.is_alive():
                print("[SYSTEM] Force terminating decoder...")
                self._decoder_process.terminate()
        
        print("[SYSTEM] All processes stopped")
    
    def run_both(self, camera_id: int = 0):
        """Запускает энкодер и декодер одновременно"""
        print("=" * 60)
        print("  Low-Bandwidth Video Streaming System")
        print("=" * 60)
        print(f"  Resolution: 128x64")
        print(f"  Target FPS: 5-10")
        print(f"  Bandwidth: ~960 bytes/sec (9600 baud)")
        print("=" * 60)
        print()
        print("Press Ctrl+C to stop")
        print()
        
        self.start_decoder()
        time.sleep(0.5)  # Даем декодеру время открыть порт
        self.start_encoder(camera_id)
        
        try:
            # Ждем завершения процессов
            while True:
                encoder_alive = self._encoder_process and self._encoder_process.is_alive()
                decoder_alive = self._decoder_process and self._decoder_process.is_alive()
                
                if not encoder_alive and not decoder_alive:
                    break
                
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted by user")
        
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Low-Bandwidth Video Streaming System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run both encoder and decoder
  %(prog)s --encoder          # Run only encoder
  %(prog)s --decoder          # Run only decoder
  %(prog)s --camera 1         # Use camera 1 instead of 0
        """
    )
    
    parser.add_argument(
        '--encoder', '-e',
        action='store_true',
        help='Run only encoder (sender)'
    )
    parser.add_argument(
        '--decoder', '-d',
        action='store_true',
        help='Run only decoder (receiver)'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera ID (default: 0)'
    )
    parser.add_argument(
        '--test-codec',
        action='store_true',
        help='Run codec test'
    )
    parser.add_argument(
        '--test-protocol',
        action='store_true',
        help='Run protocol test'
    )
    
    args = parser.parse_args()
    
    # Тесты
    if args.test_codec:
        print("Running codec test...")
        import codec
        exec(open('codec.py').read().split('if __name__')[1].split(':', 1)[1])
        return
    
    if args.test_protocol:
        print("Running protocol test...")
        import protocol
        return
    
    # Основная логика
    system = StreamingSystem()
    
    # Обработка сигналов
    def signal_handler(sig, frame):
        system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.encoder and not args.decoder:
        # Только энкодер
        from encoder import run_encoder
        run_encoder(args.camera)
    
    elif args.decoder and not args.encoder:
        # Только декодер
        from decoder import run_decoder
        run_decoder()
    
    else:
        # Оба процесса
        system.run_both(args.camera)


if __name__ == "__main__":
    main()