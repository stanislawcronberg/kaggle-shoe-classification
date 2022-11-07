from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_dataloaders(
    dataset: Dataset,
    val_size: float = 0.2,
    batch_size: int = 32,
    n_workers: int = 8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Utility function for splitting a dataset into train and validation sets.

    Args:
        dataset (Dataset): Dataset to split into train and validation sets
        val_size (float, optional): Proportion of dataset to use for validation. Defaults to 0.2.
        batch_size (int, optional): Batch size for the dataloaders. Defaults to 32.
        n_workers (int, optional): Number of workers for the dataloaders. Defaults to 8.
        random_seed (int, optional): Random seed for the dataloaders. Defaults to 42.

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
        num_workers=n_workers,
        pin_memory=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=n_workers,
        pin_memory=True,
    )

    return train_loader, validation_loader
