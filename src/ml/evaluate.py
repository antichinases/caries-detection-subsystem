from ultralytics import YOLO
from src.config import BEST_PT_PATH, DATA_YAML_PATH, IMAGE_SIZE

def evaluate_caries_model():
    """Валидация обученной модели YOLO11 на тестовой выборке (test/)."""
    if not BEST_PT_PATH.exists():
        raise FileNotFoundError(f"Веса модели не найдены: {BEST_PT_PATH}")

    model = YOLO(str(BEST_PT_PATH))
    print("[+] Запуск оценки модели на отложенной выборке test/...")

    metrics = model.val(
        data=str(DATA_YAML_PATH),
        split="test",
        imgsz=IMAGE_SIZE,
        device="cpu",
        plots=True
    )

    print("\n" + "=" * 40)
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    evaluate_caries_model()