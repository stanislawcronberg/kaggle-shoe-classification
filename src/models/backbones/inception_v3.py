import torch.nn as nn
from torchvision.models import inception_v3


class InceptionV3(nn.Module):
    def __init__(self, n_classes: int, in_channels: int):
        super().__init__()

        self.n_classes = n_classes
        self.in_channels = in_channels

        self.model = self.initialize_model()

    def forward(self, x):
        assert x.shape[2:] == (299, 299), "Input image must be of size (299, 299)"
        assert x.shape[1] == 3, "Input must have 3 channels"

        # outputs, aux = self.model(x)

        return self.model(x)

    def initialize_model(self):
        model = inception_v3(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(in_features=2048, out_features=self.n_classes).requires_grad_(True)

        return model


if __name__ == "__main__":
    model = InceptionV3(n_classes=3, in_channels=3)
