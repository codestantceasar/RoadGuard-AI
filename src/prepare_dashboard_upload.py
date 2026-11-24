
from pathlib import Path
import hashlib
import shutil
import pandas as pd

BASE_DIR = Path("data/external/roaddamagevision")
OUTPUT_DIR = Path("outputs/dashboard_upload/roaddamagevision")
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"

CLASSES = ["D00", "D10", "D20", "D40"]

SOURCE_CSV = BASE_DIR / "processed/valid.csv"

ORIGINAL_CSVS = [
    Path("data/processed/combined_train.csv"),
    Path("data/processed/validation.csv"),
    Path("data/processed/test.csv"),
    BASE_DIR / "processed/train.csv",
]


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("Preparing RoadDamageVision dashboard upload...")

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(SOURCE_CSV)

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already contains files. "
            "Move or rename it before running this script again."
        )

    df = pd.read_csv(SOURCE_CSV)

    required = ["image_path"] + CLASSES

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Hash images already used in existing datasets.
    existing_hashes = set()

    for csv_path in ORIGINAL_CSVS:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        old_df = pd.read_csv(csv_path)

        if "image_path" not in old_df.columns:
            raise ValueError(f"image_path missing in {csv_path}")

        for image_path in old_df["image_path"]:
            path = Path(image_path)

            if not path.exists():
                raise FileNotFoundError(path)

            existing_hashes.add(sha256(path))

    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    used_names = set()
    skipped_duplicates = 0

    for _, row in df.iterrows():
        source = Path(row["image_path"])

        if not source.exists():
            raise FileNotFoundError(source)

        if source.suffix.lower() not in [".jpg", ".jpeg"]:
            raise ValueError(f"Not a JPG/JPEG image: {source}")

        image_hash = sha256(source)

        if image_hash in existing_hashes:
            skipped_duplicates += 1
            continue

        filename = f"rdv_valid_{source.name}"

        if filename in used_names:
            raise ValueError(f"Duplicate filename: {filename}")

        used_names.add(filename)

        destination = IMAGE_OUTPUT_DIR / filename
        shutil.copy2(source, destination)

        labels = {}

        for class_name in CLASSES:
            value = pd.to_numeric(
                pd.Series([row[class_name]]),
                errors="coerce"
            ).iloc[0]

            if pd.isna(value) or value not in [0, 1]:
                raise ValueError(
                    f"Invalid label {class_name} for {source}"
                )

            labels[class_name] = int(value)

        records.append({
            "filename": filename,
            **labels
        })

    if not records:
        raise ValueError("No new images available for dashboard upload.")

    labels_df = pd.DataFrame(
        records,
        columns=["filename"] + CLASSES
    )

    labels_path = OUTPUT_DIR / "labels.csv"
    labels_df.to_csv(labels_path, index=False)

    # Attribution information for the dataset.
    readme = OUTPUT_DIR / "SOURCE_ATTRIBUTION.txt"

    readme.write_text(
        "Dataset: RoadDamageVision / crddc22_china_spain_uav\n"
        "Version: 2, exported May 23, 2024\n"
        "Source: https://universe.roboflow.com/"
        "luis-augusto-silva-bq4bv/crddc22_china_spain_uav\n"
        "License: CC BY 4.0\n"
        "Provided by a Roboflow user.\n"
        "Image-level labels were derived from COCO annotations.\n",
        encoding="utf-8"
    )

    total_size = sum(
        path.stat().st_size
        for path in IMAGE_OUTPUT_DIR.iterdir()
    )

    print("\n========== UPLOAD PACKAGE READY ==========")
    print(f"Images prepared: {len(records)}")
    print(f"Exact duplicates skipped: {skipped_duplicates}")
    print(f"CSV: {labels_path}")
    print(f"Images folder: {IMAGE_OUTPUT_DIR}")
    print(f"Total image size: {total_size / (1024 * 1024):.2f} MB")

    print("\nLabel totals:")

    for class_name in CLASSES:
        print(
            f"{class_name}: "
            f"{int(labels_df[class_name].sum())}"
        )

    print("\nCSV columns:", labels_df.columns.tolist())
    print("Preparation completed successfully.")


if __name__ == "__main__":
    main()
