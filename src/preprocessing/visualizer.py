import cv2
import numpy as np
from src.config import DATA_DIR, FIGURES_DIR
from src.preprocessing.filters import ImagePreprocessor


def inspect_sample():
    preprocessor = ImagePreprocessor()

    train_dir = DATA_DIR / "train" / "images"
    images = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg"))

    if not images:
        print(f"[-] В {train_dir} нет снимков для проверки.")
        return

    sample_path = images[0]
    original = preprocessor.load_image(sample_path)
    processed = preprocessor.preprocess_pipeline(sample_path)

    comparison = np.vstack((original, processed))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIGURES_DIR / f"clahe_comparison_vertical_{sample_path.stem}.png"

    cv2.imwrite(str(out_file), comparison)
    print(f"[+] Сравнительный снимок сохранен в: {out_file}")

if __name__ == "__main__":
    inspect_sample()