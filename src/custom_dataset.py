import os
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize, ToTensor
from skimage.io import imread
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_dataloaders(
        dataset,
        val_size: float = 0.2,
        batch_size: int = 32,
        random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:

    n_samples = len(dataset)
    indices = list(range(n_samples))

    train_size = n_samples - int(val_size * n_samples)

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[:train_size], indices[:train_size]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler
    )

    return train_loader, validation_loader


class FootwearDataset(Dataset):
    def __init__(self, data_dir: Path, transform=Resize((128, 128)), device=None):
        self.image_paths, self.labels = self.__initialize_filepaths_and_labels(data_dir)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = imread(str(self.image_paths[index]))
        image = torch.Tensor(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        image = image.float().to(self.device)

        label = self.labels[index]

        return image, torch.Tensor(label).to(self.device)

    @staticmethod
    def __initialize_filepaths_and_labels(data_dir: Path) -> Tuple[List[Path], np.ndarray]:
        image_paths = [image_path for image_path in data_dir.glob("**/*.jpg")]
        labels = np.array([str(path).split(os.path.sep)[-2] for path in image_paths]).reshape(-1, 1)

        encoder = OneHotEncoder(sparse=False)
        labels = encoder.fit_transform(labels)

        return image_paths, labels


if __name__ == '__main__':
    dataset = FootwearDataset(data_dir=Path("../data/"), transform=Resize((128, 128)))
    x, y = dataset[15]
    print("asdf")
