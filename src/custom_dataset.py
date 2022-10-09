import os
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
from skimage import io
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


class FootwearDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None):
        self.image_paths, self.labels = self.__initialize_filepaths_and_labels(data_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = io.imread(self.image_paths[index])
        label = self.labels[index]

        return image, label

    @staticmethod
    def __initialize_filepaths_and_labels(data_dir: Path) -> Tuple[List[Path], np.ndarray]:
        image_paths = [image_path for image_path in data_dir.glob("**/*.jpg")]
        labels = np.array([str(path).split(os.path.sep)[-2] for path in image_paths]).reshape(-1, 1)

        encoder = OneHotEncoder(sparse=False)
        labels = encoder.fit_transform(labels)

        return image_paths, labels


if __name__ == '__main__':
    dataset = FootwearDataset(data_dir=Path("../data/"))
