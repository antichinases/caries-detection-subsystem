import numpy as np
import pytest
from src.ml.predictor import CariesDetector
from src.config import ONNX_MODEL_PATH

@pytest.fixture
def detector():
    if not ONNX_MODEL_PATH.exists():
        pytest.skip("ONNX модель еще не создана. Сначала запустите src/ml/export.py")
    return CariesDetector(ONNX_MODEL_PATH)

def test_detector_output_format(detector):
    """Проверка структуры ответа детектора."""
    dummy_img = np.random.randint(0, 256, (640, 640), dtype=np.uint8)
    results = detector.predict(dummy_img, conf_thresh=0.25)

    assert isinstance(results, list)
    for item in results:
        assert set(item.keys()) == {"box", "confidence", "class_id", "label"}
        assert len(item["box"]) == 4