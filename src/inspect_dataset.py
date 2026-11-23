from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from collections import Counter

BASE_DIR = Path("data/raw/RDD2022_India/India")
XML_DIR = BASE_DIR / "train/annotations/xmls"
IMAGE_DIR = BASE_DIR / "train/images"

CLASS_NAMES = {
    "D00": "Longitudinal cracks",
    "D10": "Transverse cracks",
    "D20": "Alligator cracks",
    "D40": "Potholes",
}

# Index training images by filename stem.
image_files = {}
for path in IMAGE_DIR.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        image_files[path.stem] = path

xml_files = sorted(XML_DIR.glob("*.xml"))

class_objects = Counter()
class_images = Counter()
single_class_images = Counter()
multi_class_count = 0
empty_annotations = 0
missing_images = []
records = []

for xml_path in xml_files:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        print(f"WARNING: Could not parse {xml_path}")
        continue

    filename = root.findtext("filename")
    image_stem = Path(filename).stem if filename else xml_path.stem
    image_path = image_files.get(image_stem)

    if image_path is None:
        missing_images.append(image_stem)

    labels = []

    for obj in root.findall("object"):
        label = obj.findtext("name")
        if label:
            labels.append(label)
            class_objects[label] += 1

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        class_images[label] += 1

    if len(unique_labels) == 1:
        single_class_images[unique_labels[0]] += 1
    elif len(unique_labels) > 1:
        multi_class_count += 1
    else:
        empty_annotations += 1

    records.append({
        "image_filename": image_path.name if image_path else image_stem,
        "image_path": str(image_path) if image_path else "",
        "xml_path": str(xml_path),
        "labels": ",".join(unique_labels),
        "object_count": len(labels),
        "unique_class_count": len(unique_labels),
        "has_image": image_path is not None,
    })

# Save a detailed annotation report.
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(records)
output_file = output_dir / "india_annotation_summary.csv"
df.to_csv(output_file, index=False)

print("\n========== DATASET INSPECTION ==========")
print("Training images found:", len(image_files))
print("XML annotation files:", len(xml_files))
print("Successfully parsed annotations:", len(records))
print("Annotations without matching images:", len(missing_images))
print("Images with multiple damage classes:", multi_class_count)
print("Images with no damage objects:", empty_annotations)

print("\n========== OBJECT COUNTS ==========")
for code, description in CLASS_NAMES.items():
    print(f"{code} ({description}): {class_objects[code]} objects")

print("\n========== IMAGES PER CLASS ==========")
for code, description in CLASS_NAMES.items():
    print(f"{code} ({description}): {class_images[code]} images")

print("\n========== SINGLE-CLASS IMAGES ==========")
for code, description in CLASS_NAMES.items():
    print(f"{code} ({description}): {single_class_images[code]} images")

print("\nReport saved to:", output_file)
print("========================================")
