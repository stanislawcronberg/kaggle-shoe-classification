import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.io import imread
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Resize


def get_train_val_dataloaders(
    dataset: Dataset,
    val_size: float = 0.2,
    batch_size: int = 32,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:

    n_samples = len(dataset)
    indices = list(range(n_samples))

    train_size = n_samples - int(val_size * n_samples)

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, validation_loader


class FootwearDataset(Dataset):
    def __init__(self, data_dir: Path, transform=Resize((128, 128)), device=None):
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
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.device is not None:
            image = image.to(self.device)
            label = label.to(self.device)

        return image, label

    @staticmethod
    def __initialize_filepaths_and_labels(data_dir: Path) -> Tuple[List[Path], np.ndarray]:
        image_paths = [image_path for image_path in data_dir.glob("**/*.jpg")]
        labels = np.array([str(path).split(os.path.sep)[-2] for path in image_paths]).reshape(-1, 1)

        return image_paths, labels


if __name__ == "__main__":
    dataset = FootwearDataset(Path("data/"))
    train_loader, validation_loader = get_train_val_dataloaders(dataset)

    # Get a batch of training data
    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)

    # Visualize an image
    plt.imshow(images[0].permute(1, 2, 0).numpy().astype(np.uint8))
    label = dataset.labels_encoder.inverse_transform(labels[0].reshape(1, -1))[0][0].upper()
    plt.title(label)
    plt.show()
