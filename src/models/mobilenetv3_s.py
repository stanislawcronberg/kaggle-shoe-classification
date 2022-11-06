import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3S(nn.Module):
    def __init__(self, n_classes: int, n_channels: int = 3):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.mobilenet = self.__initialize_network()

    def forward(self, images):
        return self.mobilenet(images)

    def __initialize_network(self):
        """Initialize the MobileNetV3S network.

        Returns:
            nn.Module: MobileNetV3S network with pretrained weights

        """
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        mobilenet.features[0][0] = nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )

        for child in mobilenet.children():
            for param in child.parameters():
                param.requires_grad = False

        mobilenet.classifier[-1] = nn.Linear(in_features=1024, out_features=self.n_channels, bias=True).requires_grad_(
            True
        )

        return mobilenet.cuda()


if __name__ == "__main__":
    model = MobileNetV3S(n_classes=3)
    print(model)
