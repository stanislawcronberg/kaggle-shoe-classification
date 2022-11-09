from pathlib import Path

import hydra
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything
from torchvision.transforms import ColorJitter, Compose, RandomHorizontalFlip, Resize, ToPILImage, ToTensor

from datasets.config import ShoeCLFConfig
from datasets.shoe_dataset import FootwearDataset
from datasets.utils import get_dataloader
from models import ShoeClassifier


@hydra.main(config_path=Path("../conf"), config_name="default_config", version_base="1.2.0")
def train(cfg: ShoeCLFConfig) -> None:

    # Set seed
    if cfg.seed:
        seed_everything(cfg.seed)

    # Setup transforms
    if cfg.training.use_augmentations:
        transforms = Compose(
            [
                ToPILImage(),
                Resize(size=cfg.data.image_size),
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                ToTensor(),
            ]
        )
    else:
        transforms = Compose([ToPILImage(), Resize(cfg.data.image_size), ToTensor()])

    # Initialize train dataset
    train_dataset = FootwearDataset(index_path=cfg.data.index.train, transforms=transforms)

    # Initialize val dataset
    val_dataset = FootwearDataset(index_path=cfg.data.index.val, transforms=transforms)

    # Initialize model
    model = ShoeClassifier(cfg=cfg)

    # Initialize dataloaders
    train_loader = get_dataloader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    val_loader = get_dataloader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

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
