from pathlib import Path

# Пути проекта
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "src" / "reports" / "figures"
WEIGHTS_DIR = BASE_DIR / "weights"

# Параметры алгоритма CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Автоматическое создание папок
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Параметры нейросети YOLO11 и инференса
DATA_YAML_PATH = DATA_DIR / "data.yaml"
BASE_MODEL_NAME = "yolo11s.pt"
BEST_PT_PATH = WEIGHTS_DIR / "best.pt"
ONNX_MODEL_PATH = WEIGHTS_DIR / "best.onnx"

IMAGE_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.45

# Соответствие классов кариеса
CLASS_NAMES = {
    0: "Caries",
    1: "Deep_Caries"
}