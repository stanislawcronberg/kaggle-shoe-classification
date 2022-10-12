import torch
import time
import copy
from custom_dataset import FootwearDataset, get_train_val_dataloaders
from pathlib import Path
from torch.optim import Adam, lr_scheduler
from models import MobileNetV3S
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    dataset = FootwearDataset(data_dir=Path('../data'), device=device)

    train_loader, val_loader = get_train_val_dataloaders(dataset, val_size=0.2, batch_size=32, random_seed=42)
    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }
    dataset_sizes = {
        "train": len(train_loader),
        "val": len(val_loader)
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.argmax(labels, dim=1))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    mobilenet = MobileNetV3S(n_classes=3, n_channels=3)

    optimizer = Adam(mobilenet.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(mobilenet, criterion, optimizer, scheduler, num_epochs=25)
