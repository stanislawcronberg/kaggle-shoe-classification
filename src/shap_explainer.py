from pathlib import Path

import albumentations as A
import hydra
import numpy as np
import shap
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from datasets.config import ShoeCLFConfig
from datasets.shoe_dataset import FootwearDataset
from models import ShoeClassifier


@hydra.main(config_path=Path("../conf"), config_name="default_config", version_base="1.2.0")
def get_shap_explainer(cfg: ShoeCLFConfig):

    transforms = A.Compose([A.Resize(224, 224), ToTensorV2()])
    train_dataset = FootwearDataset(index_path="data/index/train.csv", root_data_dir=".", transforms=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    batch = next(iter(train_dataloader))
    images, _ = batch

    background = images[:20]
    test_images = images[20:24]

    model = ShoeClassifier(cfg)
    model.eval()

    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    images = test_images.numpy()

    # Print max and min values of image
    print("Max value of image: ", np.max(images))
    print("Min value of image: ", np.min(images))

    print(len(shap_values))
    print(shap_values[0].shape)

    print(images.shape)
    shap_numpy = [s.transpose(0, 2, 3, 1) for s in shap_values]
    # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    # test_numpy = ((np.swapaxes(np.swapaxes(images, 1, -1), 1, 2)) * 255).astype(np.uint8)
    test_numpy = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    # Covert the color channels of test_numpy to RGB for plotting
    # test_numpy = np.stack([test_numpy[:, :, :, 2], test_numpy[:, :, :, 1], test_numpy[:, :, :, 0]], axis=-1)
    print(test_numpy.shape)

    shap.image_plot(shap_numpy, test_numpy)


if __name__ == "__main__":
    get_shap_explainer()
