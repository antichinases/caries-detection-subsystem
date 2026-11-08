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
    """Класс инференса модели YOLO11 на CPU через ONNX Runtime."""

    def __init__(self, model_path: Path = ONNX_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX модель не найдена по пути: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 6
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]