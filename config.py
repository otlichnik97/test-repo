"""
Конфигурация системы потоковой передачи видео
для экстремально низкой пропускной способности
"""

# Разрешение видео
VIDEO_WIDTH = 128
VIDEO_HEIGHT = 64
TOTAL_PIXELS = VIDEO_WIDTH * VIDEO_HEIGHT  # 8192 пикселей

# Параметры последовательного порта
BAUD_RATE = 9600
SERIAL_PORT_TX = '/dev/ttyUSB0'
SERIAL_PORT_RX = '/dev/ttyUSB1'
SERIAL_TIMEOUT = 0.01  # 10ms таймаут для неблокирующего чтения

# Параметры протокола
PACKET_SIZE = 128  # Фиксированный размер пакета
SYNC_MAGIC = b'\xAA\xBB'
HEADER_SIZE = 5  # SYNC(2) + frame_id(1) + packet_seq(1) + data_len(1)
PAYLOAD_MAX = PACKET_SIZE - HEADER_SIZE  # 123 байта полезной нагрузки

# Ограничения пропускной способности
BYTES_PER_SECOND = 960  # ~9600 бод / 10 бит на байт
MAX_COMPRESSED_SIZE = 960  # Максимальный размер сжатого кадра (байт)
TARGET_COMPRESSED_SIZE = 250 # Целевой размер для нормальной работы
TARGET_FPS = 8  # Целевая частота кадров

# Параметры обработки изображения (Canny edge detection)
CANNY_THRESHOLD1 = 120
CANNY_THRESHOLD2 = 220
DILATE_KERNEL_SIZE = 1
DILATE_ITERATIONS = 1

# Параметры отображения
DISPLAY_SCALE = 4  # Масштаб для отображения (128x64 -> 512x256)
WINDOW_NAME_SENDER = "Sender View"
WINDOW_NAME_RECEIVER = "Receiver View"

# Отладка
DEBUG = True
SHOW_STATS = True
STATS_INTERVAL = 1.0  # Интервал вывода статистики (секунды)