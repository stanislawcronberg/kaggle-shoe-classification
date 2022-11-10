from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


class FootwearDataset(Dataset):
    def __init__(self, index_path: Union[str, Path], transform):

        # Read the index
        self.index = pd.read_csv(index_path)

        # Initialize filepaths and labels
        self.image_paths, self.labels = self.__initialize_filepaths_and_labels()

        self.transform = transform

        # Encode labels
        self.labels_encoder = OneHotEncoder(sparse=False)
        self.labels_encoder.fit(self.labels)
        self.labels = self.labels_encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image = imread(str(image_path)).astype(np.float32) / 255.0  # Read image and normalize
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and permute

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __initialize_filepaths_and_labels(self) -> tuple[list[Path], np.ndarray]:
        image_paths = self.index["image_path"].values
        labels = self.index["label"].values.reshape(-1, 1)

        return image_paths, labels
