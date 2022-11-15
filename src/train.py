from pathlib import Path

import albumentations as A
import hydra
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from lightning_lite.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from datasets.config import ShoeCLFConfig
from datasets.shoe_dataset import FootwearDataset
from models import ShoeClassifier


@hydra.main(config_path=Path("../conf"), config_name="default_config", version_base="1.2.0")
def train(cfg: ShoeCLFConfig) -> None:

    # Set seed
    if cfg.seed:
        seed_everything(cfg.seed)

    # Setup transforms
    if cfg.training.use_augmentations:
        transforms = A.Compose(
            [
                A.Resize(*cfg.data.image_size),
                A.ColorJitter(brightness=0.4, contrast=0.4, p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.CoarseDropout(max_holes=5, max_height=8, max_width=8, p=0.4),
                ToTensorV2(),
            ]
        )
    else:
        transforms = A.Compose([A.Resize(*cfg.data.image_size), ToTensorV2()])

    # Initialize training/validation datasets
    train_dataset = FootwearDataset(
        index_path=cfg.data.index.train,
        root_data_dir=cfg.data.root_data_dir,
        transforms=transforms,
    )
    val_dataset = FootwearDataset(index_path=cfg.data.index.val, root_data_dir=".", transforms=transforms)

    # Initialize training/validation dataloaders
    train_loader = DataLoader(train_dataset, **cfg.training.dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **cfg.eval.dataloader_kwargs)

    # Setup early stopping
    if cfg.training.use_early_stopping:
        early_stopping = pl.callbacks.EarlyStopping(**cfg.training.early_stopping_kwargs)

    # Initialize model & trainer
    model = ShoeClassifier(cfg=cfg)
    trainer = pl.Trainer(
        **cfg.training.trainer_kwargs,
        callbacks=[early_stopping] if early_stopping else None,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
