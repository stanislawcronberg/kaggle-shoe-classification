from pathlib import Path

import hydra
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from datasets.config import ShoeCLFConfig
from datasets.shoe_dataset import FootwearDataset
from datasets.utils import get_dataloader
from models import ShoeClassifier


@hydra.main(config_path=Path("../conf"), config_name="default_config", version_base="1.2.0")
def evaluate(cfg: ShoeCLFConfig) -> None:

    # Set seed
    if cfg.seed:
        seed_everything(cfg.seed)

    # Setup transforms for evaluation
    transforms = Compose([ToPILImage(), Resize(cfg.data.image_size), ToTensor()])

    # Initialize test dataset
    test_dataset = FootwearDataset(index_path=cfg.data.index.test, transform=transforms)

    # Initialize model
    model = ShoeClassifier(cfg=cfg)

    # Initialize dataloader
    test_loader = get_dataloader(
        dataset=test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    # Setup trainer
    trainer = pl.Trainer(**cfg.training.trainer_kwargs)

    # Evaluate model
    trainer.test(
        model=model,
        dataloaders=test_loader,
        ckpt_path=cfg.eval.ckpt_path,
    )


if __name__ == "__main__":
    evaluate()
