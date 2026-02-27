#!/usr/bin/env python3
"""
Оркестратор запуска системы
Запускает Encoder и Decoder как отдельные процессы
"""

import sys
import os
import time
import signal
import multiprocessing as mp
from multiprocessing import Process, Event


def run_encoder_process(stop_event: Event, camera_id: int = 0):
    """
    Процесс энкодера
    """
    import cv2
    import numpy as np
    import serial
    import threading
    
    from config import (
        SERIAL_PORT_TX, SERIAL_BAUDRATE,
        FRAME_WIDTH, FRAME_HEIGHT,
        CANNY_THRESHOLD_1, CANNY_THRESHOLD_2,
        DILATE_KERNEL_SIZE, DILATE_ITERATIONS,
        PACKET_SIZE, PAYLOAD_SIZE, PACKET_INTERVAL_MS,
        BUCKET_MAX_SIZE, DEBUG_PRINT_STATS, STATS_INTERVAL_SEC,
        DISPLAY_WAIT_MS
    )
    from codec import FrameEncoder
    from protocol import PacketPacker
    
    print("[ENCODER] Starting...")
    
    running = True
    cap = None
    ser = None
    
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[ENCODER] Failed to open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        ser = serial.Serial(
            port=SERIAL_PORT_TX,
            baudrate=SERIAL_BAUDRATE,
            timeout=0.01,
            write_timeout=0.1
        )
        print(f"[ENCODER] Serial port {SERIAL_PORT_TX} opened")
        
    except Exception as e:
        print(f"[ENCODER] Init failed: {e}")
        if cap:
            cap.release()
        return
    
    frame_encoder = FrameEncoder()
    packer = PacketPacker()
    
    # Буфер (leaky bucket)
    bucket_lock = threading.Lock()
    bucket_data = bytearray()
    bucket_max = BUCKET_MAX_SIZE
    
    # Статистика
    stats = {
        'frames': 0,
        'packets': 0,
        'bytes': 0,
        'dropped': 0,  # Добавлено: счетчик сброшенных данных
        'start_time': time.time()
    }
    
    dilate_kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
    
    display_frame = None
    display_lock = threading.Lock()
    
    def capture_thread_func():
        nonlocal running, display_frame, bucket_data
        
        while running and not stop_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Обработка кадра
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT), 
                                    interpolation=cv2.INTER_AREA)
                edges = cv2.Canny(resized, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
                dilated = cv2.dilate(edges, dilate_kernel, iterations=DILATE_ITERATIONS)
                binary = (dilated > 127).astype(np.uint8)
                
                stats['frames'] += 1
                
                # Кодирование
                encoded_data, frame_type, _ = frame_encoder.encode(binary)
                packer.add_frame(encoded_data, frame_type)
                
                # Перекачка в bucket
                while packer.has_data():
                    packet = packer.get_next_packet()
                    if packet:
                        with bucket_lock:
                            # ИЗМЕНЕНИЕ: агрессивный сброс старых данных
                            # Если буфер заполнен, сбрасываем ВСЁ старое
                            if len(bucket_data) + len(packet) > bucket_max:
                                dropped = len(bucket_data)
                                bucket_data.clear()
                                stats['dropped'] += dropped
                            bucket_data.extend(packet)
                
                # Для отображения
                with display_lock:
                    display_frame = (binary * 255).astype(np.uint8)
                
            except Exception as e:
                if running:
                    print(f"[ENCODER] Capture error: {e}")
                break
            
            time.sleep(0.001)
        
        print("[ENCODER] Capture thread stopped")
    
    def send_thread_func():
        nonlocal running, bucket_data
        
        interval = PACKET_INTERVAL_MS / 1000.0
        next_send = time.time()
        
        while running and not stop_event.is_set():
            try:
                now = time.time()
                
                if now >= next_send:
                    packet = None
                    with bucket_lock:
                        if len(bucket_data) >= PACKET_SIZE:
                            packet = bytes(bucket_data[:PACKET_SIZE])
                            del bucket_data[:PACKET_SIZE]
                    
                    if packet and ser and ser.is_open:
                        try:
                            ser.write(packet)
                            ser.flush()
                            stats['packets'] += 1
                            stats['bytes'] += len(packet)
                        except serial.SerialException:
                            pass
                    
                    next_send += interval
                    if next_send < now - interval:
                        next_send = now + interval
                
                sleep_time = next_send - time.time()
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.01))
                    
            except Exception as e:
                if running:
                    print(f"[ENCODER] Send error: {e}")
                break
        
        print("[ENCODER] Send thread stopped")
    
    # Запуск потоков
    capture_t = threading.Thread(target=capture_thread_func, daemon=True)
    send_t = threading.Thread(target=send_thread_func, daemon=True)
    
    capture_t.start()
    send_t.start()
    
    last_stats_time = time.time()
    window_name = "Sender View"
    window_created = False
    
    try:
        while not stop_event.is_set() and running:
            frame_show = None
            with display_lock:
                if display_frame is not None:
                    frame_show = display_frame.copy()
            
            if frame_show is not None:
                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    window_created = True
                
                display = cv2.resize(frame_show, (512, 256), 
                                    interpolation=cv2.INTER_NEAREST)
                
                with bucket_lock:
                    bucket_size = len(bucket_data)
                
                # Показываем задержку в пакетах
                delay_pkts = bucket_size // PACKET_SIZE
                delay_ms = delay_pkts * PACKET_INTERVAL_MS
                
                cv2.putText(display, f"Buf: {bucket_size}B ({delay_pkts}pkt ~{delay_ms}ms)", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
                cv2.putText(display, f"Sent: {stats['packets']}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
                cv2.putText(display, f"Drop: {stats['dropped']}B", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
                
                cv2.imshow(window_name, display)
            
            if DEBUG_PRINT_STATS and time.time() - last_stats_time > STATS_INTERVAL_SEC:
                elapsed = time.time() - stats['start_time']
                if elapsed > 0.1:
                    with bucket_lock:
                        bs = len(bucket_data)
                    print(f"[ENCODER] FPS:{stats['frames']/elapsed:.1f} "
                          f"PPS:{stats['packets']/elapsed:.1f} "
                          f"Buf:{bs}B Drop:{stats['dropped']}B")
                last_stats_time = time.time()
            
            key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
            if key == ord('q') or key == 27:
                print("[ENCODER] Exit requested")
                break
                
    except Exception as e:
        print(f"[ENCODER] Main loop error: {e}")
    
    print("[ENCODER] Stopping...")
    running = False
    
    capture_t.join(timeout=0.5)
    send_t.join(timeout=0.5)
    
    try:
        if cap:
            cap.release()
    except:
        pass
    
    try:
        if ser and ser.is_open:
            ser.close()
    except:
        pass
    
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except:
        pass
    
    print("[ENCODER] Stopped")


def run_decoder_process(stop_event: Event):
    """
    Процесс декодера
    """
    import cv2
    import numpy as np
    import serial
    import threading
    
    from config import (
        SERIAL_PORT_RX, SERIAL_BAUDRATE,
        FRAME_WIDTH, FRAME_HEIGHT, PACKET_SIZE,
        DEBUG_PRINT_STATS, STATS_INTERVAL_SEC,
        DISPLAY_WAIT_MS
    )
    from codec import FrameDecoder
    from protocol import PacketParser, StreamUnpacker
    
    print("[DECODER] Starting...")
    
    running = True
    ser = None
    
    try:
        ser = serial.Serial(
            port=SERIAL_PORT_RX,
            baudrate=SERIAL_BAUDRATE,
            timeout=0.01
        )
        print(f"[DECODER] Serial port {SERIAL_PORT_RX} opened")
    except Exception as e:
        print(f"[DECODER] Failed to open serial port: {e}")
        return
    
    packet_parser = PacketParser()
    stream_unpacker = StreamUnpacker()
    frame_decoder = FrameDecoder()
    
    display_frame = None
    display_lock = threading.Lock()
    
    stats = {
        'packets': 0,
        'frames': 0,
        'bytes': 0,
        'errors': 0,
        'start_time': time.time()
    }
    
    def receive_thread_func():
        nonlocal running, display_frame
        
        while running and not stop_event.is_set():
            try:
                if not ser or not ser.is_open:
                    break
                    
                data = ser.read(256)
                
                if not data:
                    time.sleep(0.001)
                    continue
                
                stats['bytes'] += len(data)
                
                packets = packet_parser.add_bytes(data)
                
                for packet in packets:
                    stats['packets'] += 1
                    
                    frames = stream_unpacker.add_packet(packet)
                    
                    for frame_type, frame_data in frames:
                        decoded = frame_decoder.decode(frame_data, frame_type)
                        
                        if decoded is not None:
                            stats['frames'] += 1
                            with display_lock:
                                display_frame = decoded.copy()
                        else:
                            stats['errors'] += 1
                            
            except serial.SerialException:
                break
            except Exception as e:
                if running:
                    print(f"[DECODER] Receive error: {e}")
                break
        
        print("[DECODER] Receive thread stopped")
    
    receive_t = threading.Thread(target=receive_thread_func, daemon=True)
    receive_t.start()
    
    last_stats_time = time.time()
    empty_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    window_name = "Receiver View"
    window_created = False
    
    try:
        while not stop_event.is_set() and running:
            frame_show = None
            with display_lock:
                if display_frame is not None:
                    frame_show = display_frame.copy()
            
            if frame_show is None:
                frame_show = empty_frame
            
            if not window_created:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                window_created = True
            
            display = cv2.resize(frame_show, (512, 256), 
                                interpolation=cv2.INTER_NEAREST)
            
            cv2.putText(display, f"Frames: {stats['frames']}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            cv2.putText(display, f"Packets: {stats['packets']}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            cv2.putText(display, f"Errors: {stats['errors']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            
            cv2.imshow(window_name, display)
            
            if DEBUG_PRINT_STATS and time.time() - last_stats_time > STATS_INTERVAL_SEC:
                elapsed = time.time() - stats['start_time']
                if elapsed > 0.1:
                    print(f"[DECODER] FPS:{stats['frames']/elapsed:.1f} "
                          f"PPS:{stats['packets']/elapsed:.1f} "
                          f"Err:{stats['errors']}")
                last_stats_time = time.time()
            
            key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
            if key == ord('q') or key == 27:
                print("[DECODER] Exit requested")
                break
                
    except Exception as e:
        print(f"[DECODER] Main loop error: {e}")
    
    print("[DECODER] Stopping...")
    running = False
    
    receive_t.join(timeout=0.5)
    
    try:
        if ser and ser.is_open:
            ser.close()
    except:
        pass
    
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except:
        pass
    
    print("[DECODER] Stopped")


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print("=" * 60)
    print("  Edge Video Streaming System")
    print("  Resolution: 128x64, Rate: 25 pkt/s, Baud: 38400")
    print("=" * 60)
    print()
    print("Usage:")
    print("  python run.py          - Run both encoder and decoder")
    print("  python run.py encoder  - Run encoder only")
    print("  python run.py decoder  - Run decoder only")
    print()
    print("Press 'q' or ESC in any window to exit")
    print("=" * 60)
    print()
    
    mode = "both"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    stop_event = Event()
    processes = []
    
    def signal_handler(sig, frame):
        print("\n[MAIN] Signal received, stopping...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if mode in ("both", "encoder"):
            print("[MAIN] Starting encoder process...")
            encoder_proc = Process(
                target=run_encoder_process,
                args=(stop_event, 0),
                name="Encoder"
            )
            encoder_proc.start()
            processes.append(encoder_proc)
            time.sleep(0.5)
        
        if mode in ("both", "decoder"):
            print("[MAIN] Starting decoder process...")
            decoder_proc = Process(
                target=run_decoder_process,
                args=(stop_event,),
                name="Decoder"
            )
            decoder_proc.start()
            processes.append(decoder_proc)
        
        if not processes:
            print(f"[MAIN] Unknown mode: {mode}")
            return
        
        print(f"[MAIN] Running in '{mode}' mode...")
        
        while not stop_event.is_set():
            all_dead = all(not p.is_alive() for p in processes)
            if all_dead:
                break
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted")
    
    finally:
        print("[MAIN] Cleaning up...")
        stop_event.set()
        time.sleep(0.2)
        
        for proc in processes:
            if proc.is_alive():
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)
                    if proc.is_alive():
                        proc.kill()
        
        print("[MAIN] Done")


if __name__ == "__main__":
    main()