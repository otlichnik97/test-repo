"""
Кодек для сжатия/распаковки бинарных кадров
Использует RLE (Run-Length Encoding) для 1-битных изображений

КРИТИЧНО: encoder и decoder должны иметь идентичное состояние prev_frame
после обработки каждого кадра!
"""

import numpy as np
from typing import Tuple, Optional
from config import FRAME_WIDTH, FRAME_HEIGHT, FRAME_PIXELS, FRAME_TYPE_KEY, FRAME_TYPE_DELTA


class RLECodec:
    """
    RLE кодек для бинарных изображений
    
    Формат:
    - Байт 0: начальное значение (0 или 1)
    - Байты 1+: длины серий (1-255 каждая)
    
    Важно: серии чередуются, поэтому значение не хранится для каждой серии
    """
    
    def __init__(self):
        self.prev_frame: Optional[np.ndarray] = None
        self.frame_counter = 0
        self.keyframe_interval = 10
    
    def reset(self):
        """Сброс состояния кодека"""
        self.prev_frame = None
        self.frame_counter = 0
    
    def encode_frame(self, binary_frame: np.ndarray) -> Tuple[bytes, int]:
        """
        Кодирует бинарный кадр
        
        ВАЖНО: prev_frame обновляется ПОСЛЕ кодирования,
        и должен содержать ИСХОДНЫЙ кадр (не дельту!)
        """
        # Нормализуем к 0/1
        binary_frame = (binary_frame > 0).astype(np.uint8)
        
        # Определяем тип кадра
        need_keyframe = (
            self.frame_counter % self.keyframe_interval == 0 or 
            self.prev_frame is None
        )
        
        if need_keyframe:
            # Ключевой кадр - кодируем как есть
            encoded = self._encode_rle(binary_frame.flatten())
            frame_type = FRAME_TYPE_KEY
        else:
            # Дельта-кадр - кодируем XOR с предыдущим
            delta = np.bitwise_xor(binary_frame, self.prev_frame)
            encoded_delta = self._encode_rle(delta.flatten())
            
            # Сравниваем с ключевым кадром
            encoded_key = self._encode_rle(binary_frame.flatten())
            
            if len(encoded_delta) <= len(encoded_key):
                encoded = encoded_delta
                frame_type = FRAME_TYPE_DELTA
            else:
                encoded = encoded_key
                frame_type = FRAME_TYPE_KEY
        
        # КРИТИЧНО: сохраняем ИСХОДНЫЙ кадр, не дельту!
        self.prev_frame = binary_frame.copy()
        self.frame_counter += 1
        
        return encoded, frame_type
    
    def _encode_rle(self, flat_array: np.ndarray) -> bytes:
        """
        RLE-кодирование
        
        Формат:
        [start_value: 1 byte] [run_lengths: N bytes]
        
        Серии чередуются: start_value, потом !start_value, потом снова start_value...
        """
        n = len(flat_array)
        if n == 0:
            return b'\x00'
        
        result = bytearray()
        
        # Начальное значение
        current_value = int(flat_array[0]) & 1
        result.append(current_value)
        
        i = 0
        while i < n:
            # Считаем длину текущей серии
            run_start = i
            while i < n and (int(flat_array[i]) & 1) == current_value:
                i += 1
            
            run_length = i - run_start
            
            # Записываем длину (разбиваем на части по 255)
            while run_length > 0:
                chunk = min(run_length, 255)
                result.append(chunk)
                run_length -= chunk
                
                if run_length > 0:
                    # Нужно продолжить ту же серию
                    # Добавляем нулевую серию противоположного значения
                    result.append(0)
            
            # Переключаем значение
            current_value = 1 - current_value
        
        return bytes(result)
    
    def decode_frame(self, encoded: bytes, frame_type: int) -> Optional[np.ndarray]:
        """
        Декодирует кадр
        
        ВАЖНО: prev_frame обновляется ПОСЛЕ декодирования,
        и должен содержать РЕЗУЛЬТИРУЮЩИЙ кадр!
        """
        if len(encoded) < 1:
            return None
        
        try:
            # Декодируем RLE
            decoded_flat = self._decode_rle(encoded)
            
            if decoded_flat is None:
                return None
            
            # Приводим к нужному размеру
            if len(decoded_flat) < FRAME_PIXELS:
                decoded_flat = np.pad(
                    decoded_flat, 
                    (0, FRAME_PIXELS - len(decoded_flat)),
                    mode='constant', 
                    constant_values=0
                )
            elif len(decoded_flat) > FRAME_PIXELS:
                decoded_flat = decoded_flat[:FRAME_PIXELS]
            
            frame = decoded_flat.reshape((FRAME_HEIGHT, FRAME_WIDTH))
            
            # Применяем дельту если нужно
            if frame_type == FRAME_TYPE_DELTA:
                if self.prev_frame is None:
                    # Нет предыдущего кадра - не можем декодировать дельту
                    return None
                # frame содержит дельту, применяем XOR
                frame = np.bitwise_xor(frame, self.prev_frame)
            
            # КРИТИЧНО: сохраняем РЕЗУЛЬТИРУЮЩИЙ кадр
            self.prev_frame = frame.copy()
            
            return frame
            
        except Exception as e:
            return None
    
    def _decode_rle(self, encoded: bytes) -> Optional[np.ndarray]:
        """
        RLE-декодирование
        """
        if len(encoded) < 1:
            return None
        
        result = []
        current_value = encoded[0] & 1
        
        i = 1
        while i < len(encoded):
            run_length = encoded[i]
            i += 1
            
            # Добавляем пиксели текущего значения
            for _ in range(run_length):
                result.append(current_value)
                if len(result) >= FRAME_PIXELS:
                    break
            
            if len(result) >= FRAME_PIXELS:
                break
            
            # Переключаем значение для следующей серии
            current_value = 1 - current_value
        
        return np.array(result, dtype=np.uint8)


class FrameEncoder:
    """
    Высокоуровневый энкодер кадров
    """
    
    def __init__(self):
        self.codec = RLECodec()
    
    def encode(self, binary_image: np.ndarray) -> Tuple[bytes, int, int]:
        """
        Кодирует бинарное изображение
        """
        # Нормализуем к 0/1
        binary_image = (binary_image > 0).astype(np.uint8)
        
        encoded, frame_type = self.codec.encode_frame(binary_image)
        return encoded, frame_type, self.codec.frame_counter - 1
    
    def reset(self):
        self.codec.reset()


class FrameDecoder:
    """
    Высокоуровневый декодер кадров
    """
    
    def __init__(self):
        self.codec = RLECodec()
    
    def decode(self, encoded: bytes, frame_type: int) -> Optional[np.ndarray]:
        """
        Декодирует кадр
        
        Returns:
            numpy array (64, 128) с значениями 0/255 для отображения
        """
        frame = self.codec.decode_frame(encoded, frame_type)
        if frame is not None:
            return (frame * 255).astype(np.uint8)
        return None
    
    def reset(self):
        self.codec.reset()


# === ВСТРОЕННЫЙ ТЕСТ ===
if __name__ == "__main__":
    print("Running codec self-test...")
    print()
    
    encoder = FrameEncoder()
    decoder = FrameDecoder()
    
    all_passed = True
    
    # Тест движущегося объекта
    for i in range(20):
        # Создаем кадр с движущимся объектом
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        x = 5 + (i * 3) % (FRAME_WIDTH - 20)
        y = 10 + (i * 2) % (FRAME_HEIGHT - 15)
        frame[y:y+10, x:x+15] = 1
        
        # Добавим еще один объект для сложности
        frame[5:8, 100:120] = 1
        
        # Кодируем
        encoded, frame_type, _ = encoder.encode(frame)
        type_str = "KEY  " if frame_type == FRAME_TYPE_KEY else "DELTA"
        
        # Декодируем
        decoded = decoder.decode(encoded, frame_type)
        
        if decoded is None:
            print(f"Frame {i:2d}: {type_str} - FAILED (None)")
            all_passed = False
            continue
        
        decoded_bin = (decoded > 127).astype(np.uint8)
        
        # Проверяем
        if np.array_equal(frame, decoded_bin):
            print(f"Frame {i:2d}: {type_str} size={len(encoded):4d} bytes - OK")
        else:
            diff = np.sum(np.abs(frame.astype(int) - decoded_bin.astype(int)))
            print(f"Frame {i:2d}: {type_str} size={len(encoded):4d} bytes - FAILED (diff={diff})")
            all_passed = False
    
    print()
    if all_passed:
        print("=" * 40)
        print("ALL TESTS PASSED!")
        print("=" * 40)
    else:
        print("=" * 40)
        print("SOME TESTS FAILED!")
        print("=" * 40)