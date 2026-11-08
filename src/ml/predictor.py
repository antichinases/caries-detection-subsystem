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

    def _letterbox(self, img: np.ndarray, new_shape=(640, 640)) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Масштабирование снимка с сохранением пропорций и добавлением полей."""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)