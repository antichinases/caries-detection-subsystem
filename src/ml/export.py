import shutil
from pathlib import Path
from ultralytics import YOLO
from src.config import BEST_PT_PATH, ONNX_MODEL_PATH, IMAGE_SIZE

def export_model_to_onnx():
    """Конвертация весов YOLO11 из формата PyTorch (.pt) в ONNX для работы на CPU."""
    if not BEST_PT_PATH.exists():
        raise FileNotFoundError(f"Файл весов не найден: {BEST_PT_PATH}")

    print(f"[+] Загрузка модели: {BEST_PT_PATH}")
    model = YOLO(str(BEST_PT_PATH))

    print(f"[+] Конвертация в ONNX (imgsz={IMAGE_SIZE})...")
    exported_path = model.export(
        format="onnx",
        imgsz=IMAGE_SIZE,
        dynamic=False,
        opset=12,
        simplify=True
    )

    if Path(exported_path).resolve() != ONNX_MODEL_PATH.resolve():
        shutil.move(exported_path, ONNX_MODEL_PATH)

    print(f"[+] Модель успешно сохранена в: {ONNX_MODEL_PATH}")

if __name__ == "__main__":
    export_model_to_onnx()