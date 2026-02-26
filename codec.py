"""
Оптимизированный Chain Code кодек для сжатия бинарных изображений.

Принцип работы:
1. Находим все контуры (связные белые области)
2. Для каждого контура сохраняем стартовую точку
3. Кодируем путь обхода контура через направления движения
4. Используем 3 бита на направление (8 возможных)
5. Применяем дополнительное сжатие повторяющихся направлений

Формат данных:
[num_contours: 1 байт]
[contour_1]
[contour_2]
...

Формат контура:
[start_x: 1 байт][start_y: 1 байт][flags_and_length: 2 байта][chain_data...]

flags_and_length:
  - bit 15: 1 = использовать RLE для chains, 0 = raw chains
  - bits 0-14: количество кодов направлений

chain_data (raw mode):
  - Упаковка: 8 направлений по 3 бита = 24 бита = 3 байта на 8 кодов

chain_data (RLE mode):
  - [count: 4 бита][direction: 3 бита][...] упакованы по 7 бит
"""

import numpy as np
from typing import Optional, List, Tuple
from config import VIDEO_WIDTH, VIDEO_HEIGHT, MAX_COMPRESSED_SIZE, DEBUG


class ChainCodeCodec:
    """
    Оптимизированный Chain Code кодек.
    """
    
    # 8 направлений (по часовой стрелке начиная с востока)
    # Индекс = chain code, значение = (dx, dy)
    DIRECTIONS = (
        (1, 0),    # 0: E  (восток)
        (1, 1),    # 1: SE (юго-восток)
        (0, 1),    # 2: S  (юг)
        (-1, 1),   # 3: SW (юго-запад)
        (-1, 0),   # 4: W  (запад)
        (-1, -1),  # 5: NW (северо-запад)
        (0, -1),   # 6: N  (север)
        (1, -1),   # 7: NE (северо-восток)
    )
    
    # Обратный маппинг: (dx, dy) -> code
    DIR_TO_CODE = {d: i for i, d in enumerate(DIRECTIONS)}
    
    @staticmethod
    def encode(image: np.ndarray) -> Optional[bytes]:
        """
        Кодирует бинарное изображение.
        
        Args:
            image: Бинарное изображение (H x W), значения 0 или 255
            
        Returns:
            Сжатые данные или None если превышен лимит
        """
        # Нормализуем изображение
        binary = (image > 0).astype(np.uint8)
        
        # Извлекаем контуры вручную (без OpenCV для надежности)
        contours = ChainCodeCodec._extract_contours(binary)
        
        if not contours:
            # Пустое изображение
            return b'\x00'
        
        # Ограничиваем количество контуров
        contours = contours[:255]
        
        result = bytearray()
        result.append(len(contours))
        
        for start_point, chain_codes in contours:
            encoded_contour = ChainCodeCodec._encode_contour(start_point, chain_codes)
            if encoded_contour is None:
                continue
            result.extend(encoded_contour)
        
        # Проверяем лимит
        if len(result) > MAX_COMPRESSED_SIZE:
            # Пробуем урезать контуры (оставляем самые длинные/важные)
            result = ChainCodeCodec._encode_with_limit(contours, MAX_COMPRESSED_SIZE)
            if result is None:
                return None
        
        return bytes(result)
    
    @staticmethod
    def _extract_contours(binary: np.ndarray) -> List[Tuple[Tuple[int, int], List[int]]]:
        """
        Извлекает контуры из бинарного изображения.
        Использует алгоритм трассировки границ.
        
        Returns:
            Список кортежей (start_point, chain_codes)
        """
        height, width = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        contours = []
        
        # Ищем стартовые точки контуров (белые пиксели с черным соседом слева или сверху)
        for y in range(height):
            for x in range(width):
                if binary[y, x] == 0:
                    continue
                if visited[y, x]:
                    continue
                
                # Проверяем, является ли это границей
                is_border = (x == 0 or binary[y, x-1] == 0 or 
                            y == 0 or binary[y-1, x] == 0)
                
                if is_border:
                    # Трассируем контур
                    contour = ChainCodeCodec._trace_contour(binary, x, y, visited)
                    if contour and len(contour[1]) >= 2:
                        contours.append(contour)
        
        # Сортируем по длине (длинные контуры важнее)
        contours.sort(key=lambda c: len(c[1]), reverse=True)
        
        return contours
    
    @staticmethod
    def _trace_contour(binary: np.ndarray, start_x: int, start_y: int, 
                       visited: np.ndarray) -> Optional[Tuple[Tuple[int, int], List[int]]]:
        """
        Трассирует один контур начиная с заданной точки.
        Использует алгоритм Moore-Neighbor tracing.
        """
        height, width = binary.shape
        chain_codes = []
        
        x, y = start_x, start_y
        
        # Начальное направление поиска (начинаем с запада)
        search_dir = 4
        
        # Максимальная длина контура (защита от бесконечных циклов)
        max_length = width * height
        
        first_point = (x, y)
        visited[y, x] = True
        
        for _ in range(max_length):
            # Ищем следующий пиксель контура
            found = False
            
            # Проверяем 8 направлений начиная с search_dir
            for i in range(8):
                check_dir = (search_dir + i) % 8
                dx, dy = ChainCodeCodec.DIRECTIONS[check_dir]
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < width and 0 <= ny < height and binary[ny, nx] == 1:
                    # Нашли следующий пиксель
                    chain_codes.append(check_dir)
                    x, y = nx, ny
                    visited[y, x] = True
                    
                    # Обновляем направление поиска (начинаем с противоположного - 2)
                    search_dir = (check_dir + 5) % 8
                    found = True
                    break
            
            if not found:
                # Изолированный пиксель или конец линии
                break
            
            # Проверяем, вернулись ли к началу
            if (x, y) == first_point and len(chain_codes) > 2:
                break
        
        if len(chain_codes) < 2:
            return None
        
        return (first_point, chain_codes)
    
    @staticmethod
    def _encode_contour(start_point: Tuple[int, int], 
                        chain_codes: List[int]) -> Optional[bytes]:
        """
        Кодирует один контур.
        Выбирает между raw и RLE режимом.
        """
        x, y = start_point
        
        # Проверяем координаты
        if x > 255 or y > 255:
            return None
        
        # Пробуем оба режима и выбираем лучший
        raw_data = ChainCodeCodec._encode_chains_raw(chain_codes)
        rle_data = ChainCodeCodec._encode_chains_rle(chain_codes)
        
        # Выбираем более компактный вариант
        use_rle = len(rle_data) < len(raw_data)
        chain_data = rle_data if use_rle else raw_data
        
        num_codes = len(chain_codes)
        if num_codes > 0x7FFF:
            num_codes = 0x7FFF
        
        # Формируем заголовок
        flags_and_length = num_codes | (0x8000 if use_rle else 0)
        
        result = bytearray()
        result.append(x)
        result.append(y)
        result.append((flags_and_length >> 8) & 0xFF)
        result.append(flags_and_length & 0xFF)
        result.extend(chain_data)
        
        return bytes(result)
    
    @staticmethod
    def _encode_chains_raw(chain_codes: List[int]) -> bytes:
        """
        Кодирует chain codes в сыром формате.
        8 кодов (по 3 бита) = 24 бита = 3 байта.
        """
        result = bytearray()
        
        # Упаковываем по 8 кодов в 3 байта
        for i in range(0, len(chain_codes), 8):
            chunk = chain_codes[i:i+8]
            
            # Собираем 24 бита
            bits = 0
            for j, code in enumerate(chunk):
                bits |= (code & 0x7) << (21 - j * 3)
            
            # Записываем 3 байта
            result.append((bits >> 16) & 0xFF)
            result.append((bits >> 8) & 0xFF)
            result.append(bits & 0xFF)
        
        return bytes(result)
    
    @staticmethod
    def _encode_chains_rle(chain_codes: List[int]) -> bytes:
        """
        Кодирует chain codes с RLE сжатием.
        Формат: [run_length: 4 бита][direction: 3 бита] = 7 бит
        Упаковываем по 8 пар в 7 байт.
        
        Также используем дельта-кодирование направлений.
        """
        if not chain_codes:
            return b''
        
        # Создаем RLE пары (длина, направление)
        runs = []
        current_dir = chain_codes[0]
        current_len = 1
        
        for i in range(1, len(chain_codes)):
            if chain_codes[i] == current_dir and current_len < 15:
                current_len += 1
            else:
                runs.append((current_len, current_dir))
                current_dir = chain_codes[i]
                current_len = 1
        runs.append((current_len, current_dir))
        
        # Упаковываем в биты
        result = bytearray()
        bit_buffer = 0
        bit_count = 0
        
        for run_len, direction in runs:
            # 4 бита длины (1-15 -> 0-14, 0 = 16)
            encoded_len = (run_len - 1) & 0xF
            # 3 бита направления
            encoded_dir = direction & 0x7
            
            # Добавляем 7 бит
            bits = (encoded_len << 3) | encoded_dir
            bit_buffer = (bit_buffer << 7) | bits
            bit_count += 7
            
            # Сбрасываем полные байты
            while bit_count >= 8:
                bit_count -= 8
                result.append((bit_buffer >> bit_count) & 0xFF)
                bit_buffer &= (1 << bit_count) - 1
        
        # Сбрасываем остаток
        if bit_count > 0:
            result.append((bit_buffer << (8 - bit_count)) & 0xFF)
        
        return bytes(result)
    
    @staticmethod
    def _encode_with_limit(contours: List[Tuple[Tuple[int, int], List[int]]], 
                          limit: int) -> Optional[bytes]:
        """
        Кодирует контуры с учетом лимита размера.
        Постепенно добавляет контуры пока не достигнут лимит.
        """
        result = bytearray()
        result.append(0)  # Placeholder для количества контуров
        
        count = 0
        for start_point, chain_codes in contours:
            encoded = ChainCodeCodec._encode_contour(start_point, chain_codes)
            if encoded is None:
                continue
            
            # Проверяем, поместится ли
            if len(result) + len(encoded) > limit:
                break
            
            result.extend(encoded)
            count += 1
        
        if count == 0:
            return None
        
        result[0] = count
        return bytes(result)
    
    @staticmethod
    def decode(data: bytes, width: int = VIDEO_WIDTH, 
               height: int = VIDEO_HEIGHT) -> Optional[np.ndarray]:
        """
        Декодирует сжатые данные обратно в изображение.
        
        Args:
            data: Сжатые данные
            width: Ширина изображения
            height: Высота изображения
            
        Returns:
            Бинарное изображение (0/255) или None при ошибке
        """
        if not data:
            return None
        
        try:
            result = np.zeros((height, width), dtype=np.uint8)
            
            num_contours = data[0]
            if num_contours == 0:
                return result
            
            idx = 1
            
            for _ in range(num_contours):
                if idx + 4 > len(data):
                    break
                
                # Читаем заголовок контура
                start_x = data[idx]
                start_y = data[idx + 1]
                flags_and_length = (data[idx + 2] << 8) | data[idx + 3]
                idx += 4
                
                use_rle = bool(flags_and_length & 0x8000)
                num_codes = flags_and_length & 0x7FFF
                
                if num_codes == 0:
                    continue
                
                # Декодируем chain codes
                if use_rle:
                    chain_codes, bytes_read = ChainCodeCodec._decode_chains_rle(
                        data[idx:], num_codes
                    )
                else:
                    chain_codes, bytes_read = ChainCodeCodec._decode_chains_raw(
                        data[idx:], num_codes
                    )
                
                idx += bytes_read
                
                # Рисуем контур
                ChainCodeCodec._draw_contour(result, start_x, start_y, chain_codes)
            
            return result
            
        except Exception as e:
            if DEBUG:
                print(f"[CHAIN] Decode error: {e}")
            return None
    
    @staticmethod
    def _decode_chains_raw(data: bytes, num_codes: int) -> Tuple[List[int], int]:
        """Декодирует raw chain codes."""
        codes = []
        bytes_needed = (num_codes * 3 + 7) // 8  # Округление вверх
        
        bit_pos = 0
        byte_idx = 0
        
        for _ in range(num_codes):
            if byte_idx >= len(data):
                break
            
            # Извлекаем 3 бита
            bits_in_current = 8 - bit_pos
            
            if bits_in_current >= 3:
                # Все 3 бита в текущем байте
                code = (data[byte_idx] >> (bits_in_current - 3)) & 0x7
                bit_pos += 3
                if bit_pos >= 8:
                    bit_pos = 0
                    byte_idx += 1
            else:
                # Биты разделены между байтами
                code = (data[byte_idx] & ((1 << bits_in_current) - 1)) << (3 - bits_in_current)
                byte_idx += 1
                if byte_idx < len(data):
                    remaining = 3 - bits_in_current
                    code |= (data[byte_idx] >> (8 - remaining)) & ((1 << remaining) - 1)
                    bit_pos = remaining
                else:
                    bit_pos = 0
            
            codes.append(code)
        
        return codes, (num_codes * 3 + 7) // 8
    
    @staticmethod
    def _decode_chains_rle(data: bytes, num_codes: int) -> Tuple[List[int], int]:
        """Декодирует RLE chain codes."""
        codes = []
        bit_buffer = 0
        bit_count = 0
        byte_idx = 0
        
        while len(codes) < num_codes and byte_idx < len(data):
            # Подгружаем биты
            while bit_count < 7 and byte_idx < len(data):
                bit_buffer = (bit_buffer << 8) | data[byte_idx]
                bit_count += 8
                byte_idx += 1
            
            if bit_count < 7:
                break
            
            # Извлекаем 7 бит
            bit_count -= 7
            bits = (bit_buffer >> bit_count) & 0x7F
            bit_buffer &= (1 << bit_count) - 1
            
            # Декодируем
            run_len = ((bits >> 3) & 0xF) + 1
            direction = bits & 0x7
            
            # Добавляем коды
            for _ in range(min(run_len, num_codes - len(codes))):
                codes.append(direction)
        
        return codes, byte_idx
    
    @staticmethod
    def _draw_contour(image: np.ndarray, start_x: int, start_y: int, 
                      chain_codes: List[int]):
        """Рисует контур на изображении."""
        height, width = image.shape
        x, y = start_x, start_y
        
        # Рисуем стартовую точку
        if 0 <= y < height and 0 <= x < width:
            image[y, x] = 255
        
        # Следуем по chain codes
        for code in chain_codes:
            if code < 0 or code > 7:
                continue
            
            dx, dy = ChainCodeCodec.DIRECTIONS[code]
            x, y = x + dx, y + dy
            
            if 0 <= y < height and 0 <= x < width:
                image[y, x] = 255


# =============================================================================
# ПУБЛИЧНЫЙ API
# =============================================================================

def compress(image: np.ndarray) -> Optional[bytes]:
    """
    Сжимает бинарное изображение.
    
    Args:
        image: Бинарное изображение (128x64)
        
    Returns:
        Сжатые данные или None если кадр нужно пропустить
    """
    return ChainCodeCodec.encode(image)


def decompress(data: bytes, width: int = VIDEO_WIDTH, 
               height: int = VIDEO_HEIGHT) -> Optional[np.ndarray]:
    """
    Распаковывает сжатые данные.
    
    Args:
        data: Сжатые данные
        width: Ширина изображения
        height: Высота изображения
        
    Returns:
        Распакованное изображение или None при ошибке
    """
    return ChainCodeCodec.decode(data, width, height)


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

if __name__ == "__main__":
    import cv2
    import time
    
    print("=" * 60)
    print("Chain Code Codec Benchmark")
    print("=" * 60)
    
    def test_image(name: str, img: np.ndarray):
        """Тестирует изображение."""
        print(f"\n[{name}]")
        
        white_pixels = np.sum(img > 0)
        print(f"  Pixels: {white_pixels} ({white_pixels/8192*100:.2f}%)")
        
        # Сжатие
        start = time.time()
        compressed = compress(img)
        encode_time = (time.time() - start) * 1000
        
        if compressed is None:
            print(f"  FAILED: Could not compress")
            return
        
        print(f"  Compressed: {len(compressed)} bytes")
        print(f"  Ratio: {8192/len(compressed):.1f}x (raw) / {white_pixels/8/len(compressed):.1f}x (packed)")
        
        # Декомпрессия
        start = time.time()
        decoded = decompress(compressed)
        decode_time = (time.time() - start) * 1000
        
        if decoded is None:
            print(f"  FAILED: Could not decompress")
            return
        
        # Проверка точности
        original_pixels = set(zip(*np.where(img > 0)))
        decoded_pixels = set(zip(*np.where(decoded > 0)))
        
        missing = len(original_pixels - decoded_pixels)
        extra = len(decoded_pixels - original_pixels)
        accuracy = 100 * (1 - (missing + extra) / max(1, len(original_pixels)))
        
        print(f"  Accuracy: {accuracy:.1f}% (missing: {missing}, extra: {extra})")
        print(f"  Time: encode={encode_time:.2f}ms, decode={decode_time:.2f}ms")
        
        return compressed, decoded
    
    # Тест 1: Пустое изображение
    img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
    test_image("Empty", img)
    
    # Тест 2: Одна линия
    img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
    cv2.line(img, (10, 32), (117, 32), 255, 1)
    test_image("Horizontal line", img)
    
    # Тест 3: Несколько линий
    img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
    cv2.line(img, (10, 10), (100, 50), 255, 1)
    cv2.line(img, (50, 5), (60, 60), 255, 1)
    cv2.line(img, (0, 32), (127, 32), 255, 1)
    test_image("Multiple lines", img)
    
    # Тест 4: Прямоугольник
    img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
    cv2.rectangle(img, (20, 15), (100, 50), 255, 1)
    test_image("Rectangle", img)
    
    # Тест 5: Круг
    img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
    cv2.circle(img, (64, 32), 25, 255, 1)
    test_image("Circle", img)
    
    # Тест 6: Сложная сцена
    img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (50, 30), 255, 1)
    cv2.circle(img, (90, 32), 20, 255, 1)
    cv2.line(img, (0, 50), (127, 50), 255, 1)
    cv2.line(img, (60, 0), (60, 63), 255, 1)
    result = test_image("Complex scene", img)
    
    # Тест 7: Производительность
    print("\n[Performance Test]")
    img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (50, 30), 255, 1)
    cv2.circle(img, (90, 32), 20, 255, 1)
    
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        compressed = compress(img)
    encode_total = time.time() - start
    
    start = time.time()
    for _ in range(iterations):
        decoded = decompress(compressed)
    decode_total = time.time() - start
    
    print(f"  {iterations} iterations:")
    print(f"  Encode: {encode_total:.2f}s ({encode_total/iterations*1000:.2f}ms/frame)")
    print(f"  Decode: {decode_total:.2f}s ({decode_total/iterations*1000:.2f}ms/frame)")
    print(f"  Max FPS: {iterations/max(encode_total, decode_total):.0f}")
    
    # Визуализация
    if result:
        compressed, decoded = result
        
        print("\n[Visual Test] Press any key to close")
        
        img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (50, 30), 255, 1)
        cv2.circle(img, (90, 32), 20, 255, 1)
        cv2.line(img, (0, 50), (127, 50), 255, 1)
        cv2.line(img, (60, 0), (60, 63), 255, 1)
        
        # Масштабируем для отображения
        scale = 4
        display_orig = cv2.resize(img, (VIDEO_WIDTH * scale, VIDEO_HEIGHT * scale), 
                                   interpolation=cv2.INTER_NEAREST)
        display_dec = cv2.resize(decoded, (VIDEO_WIDTH * scale, VIDEO_HEIGHT * scale), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Создаем разницу
        diff = cv2.absdiff(img, decoded)
        display_diff = cv2.resize(diff, (VIDEO_WIDTH * scale, VIDEO_HEIGHT * scale), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Комбинируем
        combined = np.hstack([display_orig, display_dec, display_diff])
        
        # Добавляем подписи
        cv2.putText(combined, f"Original", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.putText(combined, f"Decoded ({len(compressed)}B)", (VIDEO_WIDTH*scale + 10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.putText(combined, f"Diff", (VIDEO_WIDTH*scale*2 + 10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        
        cv2.imshow("Chain Code Codec Test", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()