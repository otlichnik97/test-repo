"""
Кодек для сжатия/распаковки бинарных изображений
Использует блочное кодирование + RLE
"""

import numpy as np
from config import (
    VIDEO_WIDTH, VIDEO_HEIGHT, BLOCK_SIZE,
    FRAME_TYPE_KEY, FRAME_TYPE_DELTA
)


class EdgeCodec:
    """Кодек для edge-detected бинарных изображений"""
    
    def __init__(self):
        self.prev_frame = None
        self.prev_blocks = None
        self.frame_count = 0
        
        # Размеры в блоках
        self.blocks_x = VIDEO_WIDTH // BLOCK_SIZE
        self.blocks_y = VIDEO_HEIGHT // BLOCK_SIZE
        self.total_blocks = self.blocks_x * self.blocks_y
    
    def _image_to_blocks(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Преобразует бинарное изображение в массив блоков.
        Каждый блок 4x4 = 16 бит -> сжимается в 2 байта.
        Дополнительно применяем упрощение: если > 50% пикселей активны -> весь блок = 1
        """
        blocks = np.zeros(self.total_blocks, dtype=np.uint8)
        
        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                # Извлекаем блок
                y_start = by * BLOCK_SIZE
                x_start = bx * BLOCK_SIZE
                block = binary_image[y_start:y_start+BLOCK_SIZE, 
                                     x_start:x_start+BLOCK_SIZE]
                
                # Упрощение с потерей качества: бинарное значение блока
                # 0 = пустой, 1 = есть границы (>25% пикселей активны)
                block_idx = by * self.blocks_x + bx
                active_pixels = np.sum(block > 0)
                blocks[block_idx] = 1 if active_pixels >= (BLOCK_SIZE * BLOCK_SIZE) // 4 else 0
        
        return blocks
    
    def _blocks_to_image(self, blocks: np.ndarray) -> np.ndarray:
        """Восстанавливает изображение из блоков"""
        image = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
        
        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                block_idx = by * self.blocks_x + bx
                if blocks[block_idx]:
                    y_start = by * BLOCK_SIZE
                    x_start = bx * BLOCK_SIZE
                    image[y_start:y_start+BLOCK_SIZE, 
                          x_start:x_start+BLOCK_SIZE] = 255
        
        return image
    
    def _pack_blocks_to_bits(self, blocks: np.ndarray) -> bytes:
        """Упаковывает массив блоков в битовый поток (8 блоков = 1 байт)"""
        n_bytes = (len(blocks) + 7) // 8
        result = bytearray(n_bytes)
        
        for i, block in enumerate(blocks):
            if block:
                byte_idx = i // 8
                bit_idx = i % 8
                result[byte_idx] |= (1 << bit_idx)
        
        return bytes(result)
    
    def _unpack_bits_to_blocks(self, data: bytes, n_blocks: int) -> np.ndarray:
        """Распаковывает битовый поток в массив блоков"""
        blocks = np.zeros(n_blocks, dtype=np.uint8)
        
        for i in range(min(n_blocks, len(data) * 8)):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(data) and (data[byte_idx] & (1 << bit_idx)):
                blocks[i] = 1
        
        return blocks
    
    def _rle_encode(self, data: bytes) -> bytes:
        """
        RLE кодирование для бинарных данных.
        Формат: [count, value, count, value, ...]
        count: 1-255 (0 = 256 повторений)
        """
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            current = data[i]
            count = 1
            
            while i + count < len(data) and data[i + count] == current and count < 255:
                count += 1
            
            result.append(count)
            result.append(current)
            i += count
        
        return bytes(result)
    
    def _rle_decode(self, data: bytes) -> bytes:
        """Декодирование RLE"""
        result = bytearray()
        i = 0
        
        while i + 1 < len(data):
            count = data[i]
            value = data[i + 1]
            result.extend([value] * count)
            i += 2
        
        return bytes(result)
    
    def _compute_delta(self, current_blocks: np.ndarray, prev_blocks: np.ndarray) -> bytes:
        """
        Вычисляет дельту между кадрами.
        Формат: битовая маска изменённых блоков + значения изменённых блоков
        """
        # XOR для нахождения изменений
        changes = current_blocks ^ prev_blocks
        
        # Битовая маска изменений
        change_mask = self._pack_blocks_to_bits(changes)
        
        # Значения изменённых блоков (только те, что изменились)
        changed_values = current_blocks[changes == 1]
        changed_data = self._pack_blocks_to_bits(changed_values)
        
        # Формат: [len_mask:1][mask][changed_data]
        result = bytearray()
        result.append(len(change_mask))
        result.extend(change_mask)
        result.extend(changed_data)
        
        return bytes(result)
    
    def _apply_delta(self, prev_blocks: np.ndarray, delta_data: bytes) -> np.ndarray:
        """Применяет дельту к предыдущему кадру"""
        if len(delta_data) < 1:
            return prev_blocks.copy()
        
        mask_len = delta_data[0]
        if len(delta_data) < 1 + mask_len:
            return prev_blocks.copy()
        
        change_mask = delta_data[1:1+mask_len]
        changed_data = delta_data[1+mask_len:]
        
        # Распаковываем маску изменений
        changes = self._unpack_bits_to_blocks(change_mask, self.total_blocks)
        
        # Распаковываем новые значения
        n_changed = np.sum(changes)
        new_values = self._unpack_bits_to_blocks(changed_data, n_changed)
        
        # Применяем изменения
        result = prev_blocks.copy()
        value_idx = 0
        for i in range(self.total_blocks):
            if changes[i]:
                if value_idx < len(new_values):
                    result[i] = new_values[value_idx]
                value_idx += 1
        
        return result
    
    def encode(self, binary_image: np.ndarray, force_keyframe: bool = False) -> tuple:
        """
        Кодирует бинарное изображение.
        Возвращает: (frame_type, compressed_data)
        """
        # Преобразуем в блоки
        current_blocks = self._image_to_blocks(binary_image)
        
        # Определяем тип кадра
        is_keyframe = force_keyframe or self.prev_blocks is None
        
        if is_keyframe:
            # Keyframe: полные данные блоков
            raw_data = self._pack_blocks_to_bits(current_blocks)
            compressed = self._rle_encode(raw_data)
            
            # Если RLE не помогло, используем сырые данные
            if len(compressed) >= len(raw_data):
                # Маркер что без RLE (первый байт = 0)
                final_data = b'\x00' + raw_data
            else:
                # Маркер что с RLE (первый байт = 1)
                final_data = b'\x01' + compressed
            
            frame_type = FRAME_TYPE_KEY
        else:
            # Delta frame
            delta = self._compute_delta(current_blocks, self.prev_blocks)
            compressed = self._rle_encode(delta)
            
            if len(compressed) >= len(delta):
                final_data = b'\x00' + delta
            else:
                final_data = b'\x01' + compressed
            
            frame_type = FRAME_TYPE_DELTA
        
        # Сохраняем для следующего кадра
        self.prev_blocks = current_blocks.copy()
        self.frame_count += 1
        
        return frame_type, final_data
    
    def decode(self, frame_type: int, compressed_data: bytes) -> np.ndarray:
        """
        Декодирует сжатые данные в изображение.
        """
        if len(compressed_data) < 1:
            return self._get_empty_frame()
        
        # Проверяем маркер RLE
        is_rle = compressed_data[0] == 0x01
        data = compressed_data[1:]
        
        if is_rle:
            data = self._rle_decode(data)
        
        if frame_type == FRAME_TYPE_KEY:
            # Keyframe: полные данные
            blocks = self._unpack_bits_to_blocks(data, self.total_blocks)
            self.prev_blocks = blocks.copy()
        else:
            # Delta frame
            if self.prev_blocks is None:
                return self._get_empty_frame()
            
            blocks = self._apply_delta(self.prev_blocks, data)
            self.prev_blocks = blocks.copy()
        
        return self._blocks_to_image(blocks)
    
    def _get_empty_frame(self) -> np.ndarray:
        """Возвращает пустой кадр"""
        return np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
    
    def reset(self):
        """Сбрасывает состояние кодека"""
        self.prev_frame = None
        self.prev_blocks = None
        self.frame_count = 0