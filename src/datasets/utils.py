from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Utility function for getting a dataloader.

    Args:
        dataset (Dataset): Dataset to use for the dataloader.
        batch_size (int): Batch size for the dataloader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for the dataloader.
        pin_memory (bool): Whether to pin memory for the dataloader.

    Returns:
        DataLoader: Dataloader for the dataset.
    """

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader


def get_train_val_dataloaders(
    dataset: Dataset,
    val_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 8,
    random_seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Utility function for splitting a dataset into train and validation sets.

    Args:
        dataset (Dataset): Dataset to split into train and validation sets
        val_size (float, optional): Proportion of dataset to use for validation.
        batch_size (int, optional): Batch size for the dataloaders.
        num_workers (int, optional): Number of workers for the dataloaders.
        random_seed (int, optional): Random seed for the dataloaders.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders.
    """

    n_samples = len(dataset)
    indices = list(range(n_samples))

    train_size = n_samples - int(val_size * n_samples)

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader
