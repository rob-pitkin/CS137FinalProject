import numpy as np

from loss_functions import *
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import random_split, DataLoader
from SimpleUNet import *


def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # from bean_dataset import BeanImageDataset
    #
    # trainset = BeanImageDataset("/content/drive/MyDrive/CS137_Assignment3_RobPitkin/data/train")
    # validset = BeanImageDataset("/content/drive/MyDrive/CS137_Assignment3_RobPitkin/data/validation")

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=64, shuffle=True)

    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory
    else:
        total_mem = 0

    epochs = 100
    learning_rate = 0.00001

    model = UNetModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = DiceLoss(mode='multiclass', from_logits=True)

    # Recording the loss
    train_loss = []
    val_loss = []
    val_acc = []

    for i in range(epochs):
        running_train_loss = 0.0
        for j, data in enumerate(train_loader):
            x, y = data
            ## If GPU is available, move to cuda
            if device.type == "cuda":
                x = x.to(device)
                y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                loss = loss.cpu()

            running_train_loss += np.mean(loss.data.numpy())

        running_train_loss /= train_loader.__len__()
        train_loss.append(running_train_loss)

        # validate
        running_val_loss = 0.0
        running_val_acc = 0.0
        with torch.no_grad():
            for k, data in enumerate(valid_loader):
                x, y = data
                if device.type == "cuda":
                    x = x.to(device)
                    y = y.to(device)
                output = model(x)
                loss = criterion(output, y)

                if device.type == "cuda":
                    loss = loss.cpu()

                running_val_loss += np.mean(loss.data.numpy())

                for l in range(len(x)):
                    pred = torch.nn.functional.softmax(output[l], dim=0)
                    if torch.argmax(pred) == y[l]:
                        running_val_acc += 1

            running_val_acc /= len(valid_loader.dataset.y)
            running_val_loss /= valid_loader.__len__()
            val_loss.append(running_val_loss)
            val_acc.append(running_val_acc)

        # check GPU memory if necessary
        if device.type == "cuda":
            alloc_mem = torch.cuda.memory_allocated(0)
        else:
            alloc_mem = 0

        # print out
        print(
            f"Epoch [{i + 1}]: Training Loss: {running_train_loss} Validation Loss: {running_val_loss} Accuracy: {running_val_acc}" + (
                f" Allocated/Total GPU memory: {alloc_mem}/{total_mem}" if device.type == "cuda" else ""
            ))
