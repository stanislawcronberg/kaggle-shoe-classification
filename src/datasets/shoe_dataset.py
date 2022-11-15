from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


class FootwearDataset(Dataset):
    def __init__(
        self,
        index_path: Union[str, Path],
        root_data_dir: Union[str, Path],
        transforms,
    ):

        # Read the index and setup data dir
        self.index = pd.read_csv(index_path)

        # Path to parent of the data directory
        self.root_data_dir = Path(root_data_dir)

        # Initialize filepaths and labels
        self.image_paths, self.labels = self.__initialize_filepaths_and_labels()

        self.transforms = transforms

        # Encode labels
        self.labels_encoder = OneHotEncoder(sparse=False)
        self.labels_encoder.fit(self.labels)
        self.labels = self.labels_encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image = imread(image_path).astype(np.float32) / 255.0  # Read image and normalize to [0, 1]

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.float32)

        if self.transforms:
            # Take ["image"] to get the image from the albumentations output
            image = self.transforms(image=image)["image"]

        return image, label

    def __initialize_filepaths_and_labels(self) -> tuple[list[Path], np.ndarray]:
        image_paths = self.index["image_path"].values  # .values returns a numpy array of the filepaths
        image_paths = np.array([str(self.root_data_dir / image_path) for image_path in image_paths])
        labels = self.index["label"].values.reshape(-1, 1)  # Reshape to add a dimension

        return image_paths, labels
