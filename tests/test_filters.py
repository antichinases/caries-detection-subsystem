import numpy as np
import pytest
from src.preprocessing.filters import ImagePreprocessor
from src.preprocessing.augmentations import DentalAugmentor

@pytest.fixture
def dummy_radiograph():
    """Синтетический рентгеновский снимок (640x640, полутоновый)."""
    return (
        np.random.randint(0, 256, (640, 640), dtype=np.uint8))

def test_preprocessor_shape_and_dtype(dummy_radiograph):
    """Проверка неизменности размерности и типа данных uint8 после CLAHE."""
    preprocessor = ImagePreprocessor()
    enhanced = preprocessor.apply_clahe(dummy_radiograph)

    assert enhanced.shape == dummy_radiograph.shape
    assert enhanced.dtype == np.uint8

def test_denoise_median_filter(dummy_radiograph):
    """Проверка подавления шума медианным фильтром."""
    preprocessor = ImagePreprocessor()
    denoised = preprocessor.denoise(dummy_radiograph)

    assert denoised.shape == dummy_radiograph.shape
    assert isinstance(denoised, np.ndarray)

def test_augmentor_bbox_preservation(dummy_radiograph):
    """Проверка сохранения структуры и диапазона рамок YOLO после аугментации."""
    augmentor = DentalAugmentor()
    bboxes = [[0.5, 0.5, 0.1, 0.2]]
    labels = [0]

    result = augmentor.transform(dummy_radiograph, bboxes, labels)

    assert result["image"].shape == dummy_radiograph.shape
    assert len(result["bboxes"]) == 1
    for coord in result["bboxes"][0]:
        assert 0.0 <= coord <= 1.0