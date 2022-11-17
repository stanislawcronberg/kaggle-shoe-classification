from pathlib import Path

import hydra
import pytorch_lightning as pl

from datasets.config import ShoeCLFConfig
from datasets.shoe_datamodule import FootwearDataModule
from models import ShoeClassifier


@hydra.main(config_path=Path("../conf"), config_name="default_config", version_base="1.2.0")
def evaluate(cfg: ShoeCLFConfig) -> None:

    # Initialize model
    model = ShoeClassifier(cfg=cfg)

    # Setup trainer
    trainer = pl.Trainer(**cfg.training.trainer_kwargs)

    # Evaluate model
    trainer.test(
        model=model,
        datamodule=FootwearDataModule(cfg),
        ckpt_path=cfg.eval.ckpt_path,
    )


if __name__ == "__main__":
    evaluate()
