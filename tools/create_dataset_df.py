"""Given a directory of images, create a pandas dataframe with the image paths and labels.

This larger csv file can then be split into separate train, validation, and test csv files.
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def collate_image_paths_and_labels(
    data_dir: Path,
    output_dir: Path,
    target_name: str = "label",
) -> pd.DataFrame:
    """Create a dataframe with the image paths and labels.

    Args:
        data_dir (Path): Path to the directory containing the images.
        output_dir (Path): Path to the directory where the csv file will be saved.
        target_name (str): Name of the target column for the csv file.

    Returns:
        pd.DataFrame: Dataframe with the image paths and labels.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist

    image_paths = [str(image_path) for image_path in data_dir.glob("**/*.jpg")]
    labels = [str(path).split(os.path.sep)[-2] for path in image_paths]

    df = pd.DataFrame({"image_path": image_paths, target_name: labels})

    return df


def split_dataframe_into_train_val_test(
    df: pd.DataFrame,
    output_dir: Path,
    train_size: float,
    val_size: float,
    test_size: float,
) -> tuple:
    """Split a dataframe into train, validation, and test dataframes.

    Args:
        df (pd.DataFrame): Dataframe to split.
        train_size (float): Size of the train dataframe.
        val_size (float): Size of the validation dataframe.
        test_size (float): Size of the test dataframe.
    """

    # Split df into train, validation, and test dataframes stratified by label
    train_df, val_test_df = train_test_split(df, train_size=train_size, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(
        val_test_df,
        train_size=val_size / (val_size + test_size),
        random_state=42,
        stratify=val_test_df["label"],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":

    # Create the full dataframe with image paths and labels and save it to a csv file
    full_df = collate_image_paths_and_labels(data_dir=Path("data/images"), output_dir=Path("data"))
    full_df.to_csv(Path("data/full.csv"), index=False)

    # Split the full dataframe into train, validation, and test dataframes and save them to csv files
    split_dataframe_into_train_val_test(
        dataframe=full_df,
        output_dir="data/index",
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
    )
