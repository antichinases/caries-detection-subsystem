import numpy as np
import pytest
from src.preprocessing.filters import ImagePreprocessor

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