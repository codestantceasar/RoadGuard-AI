import re
import pandas as pd
from pathlib import Path

SPLITS = {
    "train": "data/processed/train.csv",
    "validation": "data/processed/validation.csv",
    "test": "data/processed/test.csv"
}

def extract_number(path):
    match = re.search(r"India_(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def main():
    data = {}

    for split, csv_path in SPLITS.items():
        df = pd.read_csv(csv_path)
        numbers = {
            number
            for number in df["image_path"].apply(extract_number)
            if number is not None
        }
        data[split] = sorted(numbers)

        print(f"{split}: {len(numbers)} numbered images")

    print("\n========== NEARBY FILENAME AUDIT ==========")

    split_names = list(data.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            first = split_names[i]
            second = split_names[j]

            first_numbers = data[first]
            second_numbers = data[second]

            close_counts = {1: 0, 5: 0, 10: 0, 20: 0}

            for number in first_numbers:
                nearest = min(
                    (abs(number - other) for other in second_numbers),
                    default=float("inf")
                )

                for distance in close_counts:
                    if nearest <= distance:
                        close_counts[distance] += 1

            print(f"\n{first} vs {second}")

            for distance, count in close_counts.items():
                print(
                    f"Images within {distance} filename positions: "
                    f"{count}"
                )

    print("\nAUDIT COMPLETED")
    print("Note: Nearby filenames are a warning signal,")
    print("not proof that images come from the same video.")


if __name__ == "__main__":
    main()
