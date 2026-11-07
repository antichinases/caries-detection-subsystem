import albumentations as A
import cv2
import numpy as np

class DentalAugmentor:
    """Пайплайн аугментации дентальных рентгенограмм с корректным пересчетом рамок YOLO."""
    def __init__(self):
        self.pipeline = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.05,
                    rotate_limit=6,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.25
            )
        )

    def transform(self, image: np.ndarray, bboxes: list, class_labels: list) -> dict:
        """Применяет аугментации к снимку и пересчитывает координаты меток."""
        augmented = self.pipeline(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        return {
            "image": augmented["image"],
            "bboxes": augmented["bboxes"],
            "class_labels": augmented["class_labels"]
        }