import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


BASE_DIR = Path("data/external/roaddamagevision")
OUTPUT_DIR = BASE_DIR / "processed"

CLASSES = ["D00", "D10", "D20", "D40"]
SPLITS = ["train", "valid", "test"]


def convert_split(split):
    annotation_file = BASE_DIR / split / "_annotations.coco.json"

    with open(annotation_file, encoding="utf-8") as f:
        coco = json.load(f)

    categories = {
        category["id"]: category["name"]
        for category in coco["categories"]
    }

    image_labels = defaultdict(set)
    object_counts = defaultdict(int)

    for annotation in coco["annotations"]:
        category = categories[annotation["category_id"]]

        if category in CLASSES:
            image_id = annotation["image_id"]
            image_labels[image_id].add(category)
            object_counts[image_id] += 1

    records = []

    for image in coco["images"]:
        image_id = image["id"]
        filename = image["file_name"]
        image_path = BASE_DIR / split / filename

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image missing: {image_path}"
            )

        labels = image_labels.get(image_id, set())

        # Exclude Repair-only and unannotated images.
        if not labels:
            continue

        record = {
            "image_filename": filename,
            "image_path": str(image_path),
            "xml_path": "",
            "labels": ",".join(sorted(labels)),
            "object_count": object_counts[image_id],
            "unique_class_count": len(labels),
            "has_image": True,
        }

        for class_name in CLASSES:
            record[class_name] = int(class_name in labels)

        records.append(record)

    df = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{split}.csv"
    df.to_csv(output_file, index=False)

    print(f"\n{split.upper()}")
    print("Images retained:", len(df))
    print("CSV saved:", output_file)

    for class_name in CLASSES:
        print(
            f"{class_name}: {int(df[class_name].sum())} images"
        )

    return df


def main():
    print("ROADGUARD - ROAD DAMAGE VISION CONVERSION")

    for split in SPLITS:
        convert_split(split)

    print("\nCONVERSION COMPLETED")


if __name__ == "__main__":
    main()
