import cv2
import numpy as np
from pathlib import Path
from src.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE

class ImagePreprocessor:
    """Класс предварительной обработки рентгенограмм перед подачей в нейросеть."""

    def __init__(self, clip_limit: float = CLAHE_CLIP_LIMIT, tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def load_image(self, image_path: str | Path) -> np.ndarray:
        """Чтение изображения и приведение к полутоновому формату."""
        path_str = str(image_path)
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить снимок по пути: {path_str}")
        return img

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Подавление высокочастотного шума матрицы визиографа."""
        # Медианный фильтр устраняет импульсный шум, сохраняя резкость границ зуба
        return cv2.medianBlur(image, 3)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Локальное выравнивание гистограммы с ограничением контраста (CLAHE)."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self._clahe.apply(image)

    def preprocess_pipeline(self, image_path: str | Path) -> np.ndarray:
        """Полный цикл: загрузка -> фильтрация шума -> усиление контраста CLAHE."""
        raw_img = self.load_image(image_path)
        filtered_img = self.denoise(raw_img)
        enhanced_img = self.apply_clahe(filtered_img)
        return enhanced_img