import cv2
import numpy as np
from pathlib import Path
from src.preprocessing.filters import ImagePreprocessor


def inspect_sample():
    preprocessor = ImagePreprocessor()

    # Берем первый попавшийся снимок из train
    train_dir = Path("data/train/images")
    images = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg"))

    if not images:
        print("[-] В data/train/images нет снимков для проверки.")
        return

    sample_path = images[0]
    original = preprocessor.load_image(sample_path)
    processed = preprocessor.preprocess_pipeline(sample_path)

    # Склеиваем снимки по горизонтали: слева оригинал, справа результат CLAHE
    comparison = np.vstack((original, processed))

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"clahe_comparison_{sample_path.stem}.png"

    cv2.imwrite(str(out_file), comparison)
    print(f"[+] Сравнительный снимок сохранен в: {out_file}")


if __name__ == "__main__":
    inspect_sample()