from pathlib import Path

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from datasets.config import ShoeCLFConfig
from datasets.shoe_dataset import FootwearDataset


class FootwearDataModule(pl.LightningDataModule):
    def __init__(self, cfg: ShoeCLFConfig):
        super().__init__()
        self.cfg = cfg

        # * Setup preprocessing and postprocessing transforms
        # The are transforms that are applied before and after the
        # augmentation transforms
        self._pre_transforms: list = [A.Resize(*self.cfg.data.image_size)]
        self._post_transforms: list = [ToTensorV2()]

    def setup(self, stage: str = None):

        if stage == "fit" or stage is None:
            self._train_transforms = self.__initialize_transforms(
                use_augmentations=True if self.cfg.training.use_augmentations else False
            )
            self._val_transforms = self.__initialize_transforms(use_augmentations=False)

            self.train_dataset = self.__initialize_dataset(
                index_path=self.cfg.data.index.train,
                transforms=self._train_transforms,
            )
            self.val_dataset = self.__initialize_dataset(
                index_path=self.cfg.data.index.val,
                transforms=self._val_transforms,
            )

        if stage == "test" or stage is None:

            self.eval_transforms = self.__initialize_transforms(use_augmentations=False)

            self.test_dataset = self.__initialize_dataset(
                index_path=self.cfg.data.index.test,
                transforms=self.eval_transforms,
            )
            self.test_transforms = self.__initialize_transforms(use_augmentations=False)

        if stage == "predict" or stage is None:
            # TODO: Decide what to do here, test_dataset is temporary
            self.predict_transforms = self.__initialize_transforms(use_augmentations=False)

            self.predict_dataset = self.__initialize_dataset(
                index_path=self.cfg.data.index.test,
                transforms=self.predict_transforms,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.training.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.eval.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.eval.dataloader_kwargs)

    def predict_dataloader(self):
        # TODO: Decide what to do here, test_dataset is temporary
        return DataLoader(self.test_dataset, **self.cfg.eval.dataloader_kwargs)

    def __initialize_dataset(self, index_path: str, transforms: A.Compose) -> FootwearDataset:
        """Initializes the dataset from the index file with the given transforms.

        Args:
            index_path (str): Path to the index file.
            transforms (A.Compose): Albumentations Compose object with all transforms.

        Returns:
            FootwearDataset: Initialized dataset.
        """
        return FootwearDataset(
            index_path=Path(index_path),
            root_data_dir=Path(self.cfg.data.root_data_dir),
            transforms=transforms,
        )

    def __initialize_transforms(self, use_augmentations: bool) -> A.Compose:
        """Initialize transforms for albumentations.

        Args:
            use_augmentations (bool): Whether to use augmentations.

        Returns:
            A.Compose: Albumentations Compose object with all transforms.
        """
        transforms = []

        transforms += self._pre_transforms

        if use_augmentations:
            transforms += self.__get_augmentations()

        transforms += self._post_transforms

        return A.Compose(transforms)

    def __get_augmentations(self) -> list:
        """Returns a list of augmentations."""
        return [
            A.ColorJitter(brightness=0.4, contrast=0.4, p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes=5, max_height=8, max_width=8, p=0.4),
        ]
