from pathlib import Path

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from config import ShoeCLFConfig
from shoe_dataset import FootwearDataset
from torch.utils.data import DataLoader


class FootwearDataModule(pl.LightningDataModule):
    def __init__(self, cfg: ShoeCLFConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str = None):
        # Initialize transforms
        self.transforms = self.__initialize_transforms()

        # Initialize datasets
        self.train_dataset = self.__initialize_dataset(index_path=self.cfg.data.index.train)
        self.val_dataset = self.__initialize_dataset(index_path=self.cfg.data.index.val)
        self.test_dataset = self.__initialize_dataset(index_path=self.cfg.data.index.test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.training.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.eval.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.eval.dataloader_kwargs)

    def __initialize_dataset(self, index_path: str):
        # TODO: Consider making root_data_dir part of the config
        return FootwearDataset(
            index_path=index_path,
            root_data_dir=Path("."),
            transforms=self.transforms,
        )

    def __initialize_transforms(self):
        return A.Compose(
            [
                A.Resize(*self.cfg.data.image_size),
                ToTensorV2(),
            ]
        )
