#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import struct
from collections import defaultdict

PORT = "/dev/ttyUSB1"
BAUD = 19200
FRAME_W, FRAME_H = 128, 64
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

def rle_decode(data: bytes):
    bits = []
    for b in data:
        val = (b >> 7) & 1
        run = b & 0x7F
        bits.extend([val] * run)
    arr = np.array(bits[:FRAME_W*FRAME_H], dtype=np.uint8)
    return arr.reshape((FRAME_H, FRAME_W))

def sync_read(ser):
    buf = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if len(buf) > 2:
            buf = buf[-2:]
        if bytes(buf) == SYNC:
            return

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    last_frame = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    frame_buffers = defaultdict(dict)
    frame_types = {}
    frame_sizes = {}

    while True:
        sync_read(ser)
        header = ser.read(6)
        if len(header) < 6:
            continue
        frame_type, frame_id, seg_id, seg_total, crc = struct.unpack("<BBBBH", header)
        payload = ser.read(PAYLOAD_SIZE)
        if len(payload) < PAYLOAD_SIZE:
            continue
        if crc16(payload) != crc:
            frame_buffers.pop(frame_id, None)
            continue
        frame_buffers[frame_id][seg_id] = payload
        frame_types[frame_id] = frame_type
        frame_sizes[frame_id] = seg_total

        if len(frame_buffers[frame_id]) == seg_total:
            ordered = b"".join(frame_buffers[frame_id][i] for i in range(seg_total))
            decoded = rle_decode(ordered)
            if frame_type == 1:
                last_frame = decoded
            else:
                last_frame = cv2.bitwise_xor(last_frame, decoded)
            frame_buffers.pop(frame_id, None)
            frame_types.pop(frame_id, None)
            frame_sizes.pop(frame_id, None)

            view = (last_frame * 255).astype(np.uint8)
            cv2.imshow("Decoded", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

if __name__ == "__main__":
    main()
