import shutil
import torch
from ultralytics import YOLO
from src.config import (
    BASE_DIR,
    DATA_YAML_PATH,
    BASE_MODEL_NAME,
    WEIGHTS_DIR,
    IMAGE_SIZE,
)

def train_caries_detector(
    epochs: int = 40,
    batch_size: int = 16,
    imgsz: int = IMAGE_SIZE,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Пайплайн дообучения модели YOLO11s на датасете рентгенограмм."""
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {DATA_YAML_PATH}")

    print(f"[+] Инициализация модели {BASE_MODEL_NAME} на устройстве: {device.upper()}")
    model = YOLO(BASE_MODEL_NAME)

    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        patience=15,
        save=True,
        project=str(BASE_DIR / "runs" / "detect"),
        name="yolo11_caries",
        exist_ok=True,
        verbose=True,
    )

    trained_best = BASE_DIR / "runs" / "detect" / "yolo11_caries" / "weights" / "best.pt"
    if trained_best.exists():
        shutil.copy(trained_best, WEIGHTS_DIR / "best.pt")
        print(f"[+] Лучшие веса скопированы в: {WEIGHTS_DIR / 'best.pt'}")

    return results

if __name__ == "__main__":
    train_caries_detector(epochs=40, batch_size=8)