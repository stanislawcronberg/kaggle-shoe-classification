import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Adam

import models.backbones as implemented_models
from datasets.config import ShoeCLFConfig


class ShoeClassifier(pl.LightningModule):
    def __init__(self, cfg: ShoeCLFConfig):
        super().__init__()
        self.save_hyperparameters()

        # Initialize config
        self.cfg = cfg
        self.learning_rate = self.cfg.training.learning_rate

        # Initialize backbone model dynamically
        self.model = self.__initialize_model()

        # Initialize loss
        self.loss = nn.CrossEntropyLoss()

        # Initialize metrics
        self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=3, top_k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.loss(logits, labels)

        acc = self.accuracy(self.__to_int_labels(logits), self.__to_int_labels(labels))

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.loss(logits, labels)

        acc = self.accuracy(self.__to_int_labels(logits), self.__to_int_labels(labels))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.loss(logits, labels)

        acc = self.accuracy(self.__to_int_labels(logits), self.__to_int_labels(labels))

        self.log("test_loss", loss, logger=True)
        self.log("test_acc", acc, logger=True)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)

    def __initialize_model(self):
        """Initialize the model dynamically.

        Returns:
            nn.Module: Model with pretrained weights.
        """

        # Check if model name exists in models/backbones
        if not hasattr(implemented_models, self.cfg.model):
            raise ValueError(f"Model {self.cfg.model} does not exist.")

        # Initialize model dynamically
        model = getattr(implemented_models, self.cfg.model)(
            n_classes=self.cfg.data.n_classes,
            in_channels=self.cfg.data.in_channels,
        )

        return model

    def __to_int_labels(self, labels) -> torch.Tensor:
        """Converts batched one-hot encoded labels or output logits to integer labels.

        # TODO: Double check output shape

        Args:
            labels (torch.Tensor): Batched one-hot encoded labels or output logits with shape (batch_size, n_classes).

        Returns:
            torch.Tensor: Integer labels with shape (batch_size,1).
        """
        return torch.argmax(labels, dim=1).int()
