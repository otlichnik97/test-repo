#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import struct
import time
from collections import deque

PORT = "/dev/ttyUSB0"
BAUD = 19200
FRAME_W, FRAME_H = 128, 64
FPS = 8
INTRA_PERIOD = 10
PACKET_SIZE = 64
PAYLOAD_SIZE = 56
SYNC = b"\xA5\x5A"

def crc16(data: bytes):
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(resized, 40, 120)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    binary = (dilated > 0).astype(np.uint8)
    return binary

def rle_encode(bits: np.ndarray):
    flat = bits.flatten()
    out = bytearray()
    current = int(flat[0])
    run = 1
    for bit in flat[1:]:
        b = int(bit)
        if b == current and run < 127:
            run += 1
        else:
            out.append((current << 7) | run)
            current = b
            run = 1
    out.append((current << 7) | run)
    return bytes(out)

def build_packets(encoded: bytes, frame_id: int, frame_type: int):
    segments = [encoded[i:i+PAYLOAD_SIZE] for i in range(0, len(encoded), PAYLOAD_SIZE)]
    total = len(segments)
    packets = []
    for sid, payload in enumerate(segments):
        payload = payload.ljust(PAYLOAD_SIZE, b"\x00")
        crc = crc16(payload)
        header = SYNC + bytes([frame_type, frame_id, sid, total]) + struct.pack("<H", crc)
        packets.append(header + payload)
    return packets

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    ser = serial.Serial(PORT, BAUD, timeout=0)
    prev = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    frame_id = 0
    next_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        binary = preprocess(frame)
        if frame_id % INTRA_PERIOD == 0:
            payload = binary
            frame_type = 1
        else:
            payload = cv2.bitwise_xor(binary, prev)
            frame_type = 0
        encoded = rle_encode(payload)
        packets = build_packets(encoded, frame_id, frame_type)
        for pkt in packets:
            ser.write(pkt)
        prev = binary
        frame_id = (frame_id + 1) % 256

        next_frame_time += 1.0 / FPS
        sleep_time = next_frame_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
