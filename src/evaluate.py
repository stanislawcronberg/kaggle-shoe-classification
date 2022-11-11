from pathlib import Path

import hydra
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from datasets.config import ShoeCLFConfig
from datasets.shoe_dataset import FootwearDataset
from models import ShoeClassifier


@hydra.main(config_path=Path("../conf"), config_name="default_config", version_base="1.2.0")
def evaluate(cfg: ShoeCLFConfig) -> None:

    # Set seed
    if cfg.seed:
        seed_everything(cfg.seed)

    # Setup transforms for evaluation
    transforms = Compose([ToPILImage(), Resize(cfg.data.image_size), ToTensor()])

    # Initialize test dataset
    test_dataset = FootwearDataset(index_path=cfg.data.index.test, transforms=transforms)

    # Initialize model
    model = ShoeClassifier(cfg=cfg)

    # Initialize dataloader
    test_loader = DataLoader(test_dataset, **cfg.eval.dataloader_kwargs)

    # Setup trainer
    trainer = pl.Trainer(**cfg.training.trainer_kwargs)

    # Evaluate model
    trainer.test(
        model=model,
        test_dataloaders=test_loader,
        ckpts_path=cfg.eval.ckpt_path,
    )


if __name__ == "__main__":
    evaluate()
