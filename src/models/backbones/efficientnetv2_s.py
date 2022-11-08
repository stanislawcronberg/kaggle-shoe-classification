import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class EffNetV2S(nn.Module):
    def __init__(self, n_classes, in_channels):
        super().__init__()
        assert in_channels == 3, "EfficientNetV2S only supports 3 input channels"

        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=n_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = EffNetV2S(3, 3)
    print(model)
