import os
from pathlib import Path

import numpy as np
import torch
from skimage.io import imread
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


class FootwearDataset(Dataset):
    def __init__(self, data_dir: Path, transform, device=None):
        self.image_paths, self.labels = self.__initialize_filepaths_and_labels(data_dir)
        self.transform = transform
        self.device = device

        # Encode labels
        self.labels_encoder = OneHotEncoder(sparse=False)
        self.labels_encoder.fit(self.labels)
        self.labels = self.labels_encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image = imread(str(image_path))

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.device is not None:
            image = image.to(self.device)
            label = label.to(self.device)

        return image, label

    @staticmethod
    def __initialize_filepaths_and_labels(data_dir: Path) -> tuple[list[Path], np.ndarray]:
        image_paths = [image_path for image_path in data_dir.glob("**/*.jpg")]
        labels = np.array([str(path).split(os.path.sep)[-2] for path in image_paths]).reshape(-1, 1)

        return image_paths, labels
