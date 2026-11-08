from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from src.config import (
    ONNX_MODEL_PATH,
    IMAGE_SIZE,
    DEFAULT_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    CLASS_NAMES,
)

class CariesDetector:
    pass
