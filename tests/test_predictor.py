import pytest
from src.ml.predictor import CariesDetector
from src.config import ONNX_MODEL_PATH

@pytest.fixture
def detector():
    if not ONNX_MODEL_PATH.exists():
        pytest.skip("ONNX модель еще не создана. Сначала запустите src/ml/export.py")
    return CariesDetector(ONNX_MODEL_PATH)
