from pathlib import Path

# Пути проекта
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Параметры алгоритма CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Соответствие классов кариеса
CLASS_NAMES = {
    0: "Caries",
    1: "Deep_Caries"
}