from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


INPUT_CSV = Path("data/processed/india_annotation_summary.csv")
OUTPUT_DIR = Path("data/processed")

CLASSES = ["D00", "D10", "D20", "D40"]
RANDOM_STATE = 42


def prepare_dataset():
    print("========== ROADGUARD PREPROCESSING ==========")

    df = pd.read_csv(INPUT_CSV)

    # Keep existing images.
    df = df[df["has_image"] == True].copy()

    df = df[
        df["image_path"].apply(
            lambda path: Path(path).is_file()
        )
    ].copy()

    # Convert selected categories into binary labels.
    for label in CLASSES:
        df[label] = df["labels"].fillna("").apply(
            lambda value: int(
                label in [
                    item.strip()
                    for item in str(value).split(",")
                    if item.strip()
                ]
            )
        )

    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid images found.")

    X = df["image_path"].values
    y = df[CLASSES].values

    # First split: 70% training, 30% temporary.
    first_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    train_idx, temporary_idx = next(
        first_split.split(X, y)
    )

    train_df = df.iloc[train_idx].copy()
    temporary_df = df.iloc[temporary_idx].copy()

    # Second split: 15% validation, 15% testing overall.
    second_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    temp_y = temporary_df[CLASSES].values

    val_idx, test_idx = next(
        second_split.split(
            temporary_df["image_path"].values,
            temp_y
        )
    )

    val_df = temporary_df.iloc[val_idx].copy()
    test_df = temporary_df.iloc[test_idx].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\n========== DATASET SIZE ==========")
    print("Total:", len(df))
    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Testing:", len(test_df))

    print("\n========== CLASS DISTRIBUTION ==========")

    for name, split in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df)
    ]:
        print(f"\n{name}:")

        for label in CLASSES:
            print(f"{label}: {int(split[label].sum())}")

        print(
            "Images with none of the four labels:",
            int((split[CLASSES].sum(axis=1) == 0).sum())
        )

    print("\nSaved train.csv, validation.csv and test.csv.")
    print("PREPROCESSING COMPLETE")


if __name__ == "__main__":
    prepare_dataset()
