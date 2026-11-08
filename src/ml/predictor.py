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

    def predict(self, image: np.ndarray, conf_thresh: float = DEFAULT_CONF_THRESHOLD) -> list[dict]:
        """Поиск очагов кариеса на снимке."""
        orig_h, orig_w = image.shape[:2]

        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_img, ratio, (pad_w, pad_h) = self._letterbox(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        blob = input_img.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = self.session.run(self.output_names, {self.input_name: blob})
        predictions = np.squeeze(outputs[0]).T

        boxes, scores, class_ids = [], [], []

        for pred in predictions:
            cls_scores = pred[4:]
            max_score = float(np.max(cls_scores))
            if max_score >= conf_thresh:
                class_id = int(np.argmax(cls_scores))
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

                x1 = cx - w / 2
                y1 = cy - h / 2
                boxes.append([x1, y1, w, h])
                scores.append(max_score)
                class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, NMS_IOU_THRESHOLD)
        detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = boxes[i]
                x1 = int(np.clip((bx - pad_w) / ratio, 0, orig_w))
                y1 = int(np.clip((by - pad_h) / ratio, 0, orig_h))
                x2 = int(np.clip((bx + bw - pad_w) / ratio, 0, orig_w))
                y2 = int(np.clip((by + bh - pad_h) / ratio, 0, orig_h))

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": round(scores[i], 3),
                    "class_id": class_ids[i],
                    "label": CLASS_NAMES.get(class_ids[i], "Unknown")
                })

        return detections