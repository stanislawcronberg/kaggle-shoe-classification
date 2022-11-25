from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from tqdm import tqdm

from datasets.shoe_dataset import FootwearDataset
from datasets.utils import get_dataloader
from models import MobileNetV3S

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)


if __name__ == "__main__":

    # Setup transforms
    transforms = Compose([ToPILImage(), Resize(size=(128, 128)), ToTensor()])

    # Setup train and validation datasets from the index csv files
    train_data = FootwearDataset(index_path=Path("data/index/train.csv"), transform=transforms)
    val_data = FootwearDataset(index_path=Path("data/index/val.csv"), transform=transforms)
    test_data = FootwearDataset(index_path=Path("data/index/test.csv"), transform=transforms)

    # Setup training hyperparameters
    num_epochs = 1
    learning_rate = 0.01
    batch_size = 128

    # Setup train and validation dataloaders from the datasets with get_data_loader utility function
    train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = get_dataloader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=None)
    test_loader = get_dataloader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=None)

    # Setup model, loss function, and optimizer
    model = MobileNetV3S(n_classes=3, in_channels=3).to(device)  # Note: We move the model to the device
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)  # Note: We pass the model parameters to the optimizer

    """Training loop of the model.

    We use the tqdm library to create a progress bar for the training loop.
    Using plain PyTorch, we have to implement the training loop ourselves.

    We have to call model.train() before training and model.eval() before evaluation.
    By default the model mode is set to train, so we don't have to call model.train() before training.
    """

    # Inspect what mode the model is in
    print("Model mode:", "train" if model.training else "eval")

    for epoch in range(num_epochs):  # 1 iteration = 1 epoch

        model.train()  # Set model to train mode

        running_loss = 0.0
        for i, data in enumerate(pbar := tqdm(train_loader), 0):
            # Get images and labels from the dataloader
            images, labels = data

            # Push images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            """
            We call optimizer.zero_grad() to reset the gradients of the model parameters
            before computing the gradients of the current batch
            Otherwise, the gradients will be accumulated
            (i.e. the gradients of the current batch will be added to the gradients of the previous batch)

            This is not what we want because we want to compute the gradients of the current batch only
            We call optimizer.zero_grad() before the forward pass
            """

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and gradient descent update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss every 10 mini-batches
            running_loss += loss.item()
            if i % 10 == 0:
                pbar.set_description(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():  # We don't need to compute gradients in the validation loop
            for i, data in enumerate(pbar := tqdm(val_loader), 0):

                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                val_loss = criterion(outputs, labels)

                running_val_loss += val_loss.item()
                if i % 10 == 0:
                    pbar.set_description(f"[{epoch + 1}, {i + 1:5d}] val_loss: {running_val_loss / 10:.3f}")
                    running_val_loss = 0.0

    """Evaluation of the model.

    We use the torch.no_grad() context manager to tell PyTorch that we don't need to track the gradients.

    Also we call model.eval() to tell PyTorch that we are in evaluation mode. This is important because some layers
    like dropout or batch normalization behave differently during training and evaluation.

    - We don't want to use dropout during evaluation.
    - Batch normalization doesn't use the running mean and variance during training.
    """
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):

            # Get inputs and labels from the dataloader and push them to the device
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # Convert one-hot encoded labels to class indices
            labels = torch.argmax(labels, dim=1)

            # calculate outputs by running images through the network
            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            predicted = torch.argmax(outputs.data, dim=1)

            total += labels.size(0)  # Increment total by the batch size (first dimension of the tensor)
            correct += (predicted == labels).sum().item()  # item() extracts the value from the tensor

    model.train()  # Not necessary, but good practice to set the model back to train mode

    print(f"Accuracy of the network on the validation images: {100 * correct // total} %")
