"""
Конфигурация системы потоковой передачи видео
"""

# Разрешение видео
VIDEO_WIDTH = 128
VIDEO_HEIGHT = 64
TOTAL_PIXELS = VIDEO_WIDTH * VIDEO_HEIGHT  # 8192 пикселей = 1024 байта в сыром виде

# Параметры Canny edge detection
CANNY_THRESHOLD_1 = 120
CANNY_THRESHOLD_2 = 220

# Параметры дилатации
DILATE_KERNEL_SIZE = 1
DILATE_ITERATIONS = 1

# Параметры кодека
KEYFRAME_INTERVAL = 10  # Каждые 10 кадров - keyframe
BLOCK_SIZE = 4  # Размер блока для блочного кодирования (4x4)

# Параметры протокола
SYNC_MAGIC = b'\xAA\xBB'
PACKET_SIZE = 128  # Строго 128 байт
HEADER_SIZE = 5  # SYNC(2) + frame_id(1) + frame_type(1) + packet_seq(1)
PAYLOAD_SIZE = PACKET_SIZE - HEADER_SIZE  # 123 байта полезной нагрузки

# Типы кадров
FRAME_TYPE_KEY = 0x01
FRAME_TYPE_DELTA = 0x02
FRAME_TYPE_CONTINUATION = 0x03  # Продолжение данных кадра

# Параметры последовательного порта
SERIAL_PORT_TX = '/dev/ttyUSB0'
SERIAL_PORT_RX = '/dev/ttyUSB1'
BAUD_RATE = 38400

# Параметры потока
TARGET_PACKETS_PER_SECOND = 25  # ~25 пакетов/сек
PACKET_INTERVAL = 1.0 / TARGET_PACKETS_PER_SECOND  # 40ms между пакетами
TARGET_FPS = 10  # Целевой FPS (между 5-10)

# Параметры буферизации
MAX_PACKETS_PER_FRAME = 12  # Максимум пакетов на кадр (с запасом)
FRAME_TIMEOUT = 0.5  # Таймаут сборки кадра в секундах