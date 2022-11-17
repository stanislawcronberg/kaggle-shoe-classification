from pathlib import Path

import hydra
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

from datasets.config import ShoeCLFConfig
from datasets.shoe_datamodule import FootwearDataModule
from models import ShoeClassifier


@hydra.main(config_path=Path("../conf"), config_name="default_config", version_base="1.2.0")
def train(cfg: ShoeCLFConfig) -> None:

    # Set seed
    if cfg.seed:
        seed_everything(cfg.seed)

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
    trainer.fit(model, datamodule=FootwearDataModule(cfg=cfg))


if __name__ == "__main__":
    train()
