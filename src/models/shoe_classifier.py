from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Adam
from torchvision.transforms import (
    ColorJitter,
    Compose,
    GaussianBlur,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
    ToPILImage,
    ToTensor,
)

from datasets.config import ShoeCLFConfig
from datasets.shoe_dataset import FootwearDataset
from datasets.utils import get_train_val_dataloaders


class ShoeClassifier(pl.LightningModule):
    def __init__(self, cfg: ShoeCLFConfig, learning_rate: float = None):
        super().__init__()

        # Initialize config
        self.cfg = cfg

        # Learning rate for automatic learning rate finder
        self.learning_rate = learning_rate

        # Initialize backbone model dynamically
        self.model = self.__initialize_model()

        self.loss = nn.CrossEntropyLoss()

        # Initialize metrics
        self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=3, top_k=1)

        # Save model hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.loss(logits, labels)

        preds = torch.argmax(logits, dim=1).int()
        int_labels = torch.argmax(labels, dim=1).int()

        acc = self.accuracy(preds, int_labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.loss(logits, labels)

        preds = torch.argmax(logits, dim=1).int()
        int_labels = torch.argmax(labels, dim=1).int()
        acc = self.accuracy(preds, int_labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss(logits, labels)

        preds = torch.argmax(logits, dim=1).int()
        int_labels = torch.argmax(labels, dim=1).int()
        acc = self.accuracy(preds, int_labels)

        self.log("test_loss", loss, logger=True)
        self.log("test_acc", acc, logger=True)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)

    def __initialize_model(self):
        """Initialize the model dynamically.

        Returns:
            nn.Module: Model with pretrained weights.
        """

        implemented_models = __import__("models.backbones")

        # Check if model name exists in models/backbones
        if not hasattr(implemented_models, self.cfg.model):
            raise ValueError(f"Model {self.cfg.model} does not exist.")

        # Initialize model dynamically
        model = getattr(implemented_models, self.cfg.model)(
            n_classes=self.cfg.data.n_classes,
            in_channels=self.cfg.data.in_channels,
        )

        return model


if __name__ == "__main__":
    model = ShoeClassifier()
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10, precision=16)
    dataset = FootwearDataset(
        Path("data/"),
        transform=Compose(
            [
                ToPILImage(),
                Resize((240, 240)),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.3),
                RandomRotation(degrees=10),
                GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 2.0)),
                ToTensor(),
            ]
        ),
    )
    train_dataloader, val_dataloader = get_train_val_dataloaders(dataset)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
