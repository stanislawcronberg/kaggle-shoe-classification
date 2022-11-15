from pathlib import Path

import albumentations as A
import hydra
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from datasets.config import ShoeCLFConfig
from datasets.shoe_dataset import FootwearDataset
from models import ShoeClassifier


@hydra.main(config_path=Path("../conf"), config_name="default_config", version_base="1.2.0")
def evaluate(cfg: ShoeCLFConfig) -> None:

    # Setup transforms for evaluation
    transforms = A.Compose([A.Resize(*cfg.data.image_size), ToTensorV2()])

    # Initialize test dataset and dataloader
    test_dataset = FootwearDataset(
        index_path=cfg.data.index.test,
        root_data_dir=cfg.data.root_data_dir,
        transforms=transforms,
    )
    test_loader = DataLoader(test_dataset, **cfg.eval.dataloader_kwargs)

    # Initialize model
    model = ShoeClassifier(cfg=cfg)

    # Setup trainer
    trainer = pl.Trainer(**cfg.training.trainer_kwargs)

    # Evaluate model
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=cfg.eval.ckpt_path)


if __name__ == "__main__":
    evaluate()
