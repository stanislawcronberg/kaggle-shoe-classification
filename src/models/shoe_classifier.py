import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Adam

from datasets.config import ShoeCLFConfig


class ShoeClassifier(pl.LightningModule):
    def __init__(self, cfg: ShoeCLFConfig):
        super().__init__()

        # Initialize config
        self.cfg = cfg

        # Initialize backbone model dynamically
        self.model = self.__initialize_model()

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

        # Initialize loss
        self.loss = nn.CrossEntropyLoss()

        # Initialize metrics
        self.accuracy = torchmetrics.Accuracy(num_classes=3, task="multiclass", top_k=1)

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
        return Adam(self.model.parameters(), lr=self.cfg.training.learning_rate)

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
