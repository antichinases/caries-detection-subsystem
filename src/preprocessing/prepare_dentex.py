import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Определение путей относительно корня проекта
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "training_data" / "training_data" / "quadrant-enumeration-disease"
OUTPUT_DIR = BASE_DIR / "data"
JSON_PATH = RAW_DIR / "train_quadrant_enumeration_disease.json"

# Маппинг классов из category_id_3 в формат YOLO (0 и 1)
# 1 -> Caries (класс 0)
# 3 -> Deep Caries (класс 1)
DISEASE_MAPPING = {
    1: 0,
    3: 1
}

def convert_dentex():
    if not JSON_PATH.exists():
        print(f"[-] Файл аннотаций не найден: {JSON_PATH}")
        return

    print(f"[+] Загрузка аннотаций из {JSON_PATH.name}...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Индексируем метаданные изображений по их id
    images_meta = {img["id"]: img for img in coco_data.get("images", [])}

    # Словарь для сбора меток YOLO: {image_id: [строки YOLO]}
    img_annotations = {img["id"]: [] for img in coco_data.get("images", [])}

    total_caries_boxes = 0

    #Парсинг аннотаций с фильтрацией по category_id_3
    for ann in coco_data.get("annotations", []):
        disease_id = ann.get("category_id_3")

        # Оставляем только целевые патологии кариеса
        if disease_id in DISEASE_MAPPING:
            yolo_class = DISEASE_MAPPING[disease_id]
            img_id = ann["image_id"]
            img_info = images_meta[img_id]

            img_w = img_info["width"]
            img_h = img_info["height"]
            bbox = ann["bbox"]  # [x_min, y_min, width, height]

            # Перевод в нормализованный формат YOLO [x_center, y_center, w, h] (0.0 - 1.0)
            x_center = (bbox[0] + bbox[2] / 2.0) / img_w
            y_center = (bbox[1] + bbox[3] / 2.0) / img_h
            norm_w = bbox[2] / img_w
            norm_h = bbox[3] / img_h

            # Ограничение значений в диапазоне [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            yolo_line = f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            img_annotations[img_id].append(yolo_line)
            total_caries_boxes += 1

    # Отбираем только снимки, на которых найден хотя бы один очаг кариеса
    valid_image_ids = [img_id for img_id, bboxes in img_annotations.items() if len(bboxes) > 0]
    print(f"[+] Всего размечено рамок кариеса: {total_caries_boxes}")
    print(f"[+] Снимков с наличием кариеса: {len(valid_image_ids)}")

    if not valid_image_ids:
        print("[-] Не найдено ни одного объекта с целевыми классами.")
        return

    # 4. Разбиение выборки: 70% Train, 20% Valid, 10% Test
    train_ids, test_val_ids = train_test_split(valid_image_ids, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(test_val_ids, test_size=0.333, random_state=42)

    splits = {
        "train": train_ids,
        "valid": val_ids,
        "test": test_ids
    }

    # Поиск исходных картинок внутри RAW_DIR
    def find_image_file(file_name):
        direct = RAW_DIR / file_name
        if direct.exists():
            return direct
        # Рекурсивный поиск, если файлы лежат во вложенной папке
        matches = list(RAW_DIR.rglob(file_name))
        return matches[0] if matches else None

    # 5. Копирование файлов и сохранение текстовых меток
    for split_name, ids in splits.items():
        img_out = OUTPUT_DIR / split_name / "images"
        lbl_out = OUTPUT_DIR / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img_id in ids:
            file_name = images_meta[img_id]["file_name"]
            src_file = find_image_file(file_name)

            if src_file and src_file.exists():
                shutil.copy(src_file, img_out / file_name)

                txt_filename = Path(file_name).stem + ".txt"
                with open(lbl_out / txt_filename, "w", encoding="utf-8") as f_lbl:
                    f_lbl.write("\n".join(img_annotations[img_id]))
                copied += 1

        print(f"[+] Выборка '{split_name}': успешно сохранено {copied} снимков и меток.")

    print("\n[+] Конвертация успешно завершена!")

if __name__ == "__main__":
    convert_dentex()