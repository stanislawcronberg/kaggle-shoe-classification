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

from src.datasets.shoe_dataset import FootwearDataset
from src.datasets.utils import get_train_val_dataloaders
from src.models import MobileNetV3S


class ShoeClassifier(pl.LightningModule):
    def __init__(self, n_classes: int = 3):
        super().__init__()

        self.model = MobileNetV3S(n_classes=n_classes, in_channels=3)
        self.loss = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(num_classes=n_classes, task="multiclass", top_k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.loss(logits, labels)

        preds = torch.argmax(logits, dim=1).int()
        int_labels = torch.argmax(labels, dim=1).int()

        acc = self.accuracy(preds, int_labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.loss(logits, labels)

        preds = torch.argmax(logits, dim=1).int()
        int_labels = torch.argmax(labels, dim=1).int()
        acc = self.accuracy(preds, int_labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-3)


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
