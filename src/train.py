import torch
from custom_dataset import FootwearDataset, get_train_val_dataloaders
from pathlib import Path
from torch.optim import Adam
from models import MobileNetV3S
import torch.nn as nn
from torchvision.transforms import Resize

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":

    dataset = FootwearDataset(data_dir=Path("data"), transform=Resize((128, 128)), device=device)
    train_loader, val_loader = get_train_val_dataloaders(dataset=dataset, val_size=0.2)

    mobilenet = MobileNetV3S(n_classes=3, n_channels=3)

    optimizer = Adam(mobilenet.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 20

    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mobilenet(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    # evaluate
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            labels = labels.to(device)
            labels = torch.argmax(labels, dim=1)
            images = images.to(device)
            # calculate outputs by running images through the network
            outputs = mobilenet(images)
            # the class with the highest energy is what we choose as prediction
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the validation images: {100 * correct // total} %')
