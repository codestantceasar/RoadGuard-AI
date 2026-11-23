from pathlib import Path

import pandas as pd
import imagehash
from PIL import Image

IMAGE_DIR = Path(
    "data/raw/RDD2022_India/India/train/images"
)

SAMPLE_NUMBERS = [
    1000, 1001, 1002, 1003, 1004,
    3000, 3001, 3002, 3003, 3004,
    6389, 6390, 6391, 6392, 6393
]


def main():
    print("========== IMAGE SEQUENCE INSPECTION ==========")

    records = []

    for number in SAMPLE_NUMBERS:
        filename = f"India_{number:06d}.jpg"
        path = IMAGE_DIR / filename

        if not path.exists():
            print("Missing:", filename)
            continue

        with Image.open(path) as image:
            image_hash = imagehash.phash(image)

        records.append({
            "filename": filename,
            "number": number,
            "hash": image_hash
        })

    for i in range(len(records) - 1):
        first = records[i]
        second = records[i + 1]

        if second["number"] - first["number"] != 1:
            continue

        distance = first["hash"] - second["hash"]

        print(
            f'{first["filename"]} vs '
            f'{second["filename"]}: '
            f'Hash distance = {distance}'
        )

    print("\nINSPECTION COMPLETED")


if __name__ == "__main__":
    main()
