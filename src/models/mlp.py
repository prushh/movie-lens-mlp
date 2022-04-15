import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset, DataLoader

from src.data.dataset import MovieDataset
from src.utils.const import SEED


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()

        dropout = 0.2
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

        print(self.model)

    def forward(self, x):
        return self.model(x)


def train_model(model, criterion, optimizer, epochs, data_loader, device):
    model.train()
    loss_values = []
    for epoch in range(epochs):
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            # print(f'data: {data}\ntargets: {targets}')

            optimizer.zero_grad()

            # Forward pass
            y_pred = model(data)

            # print(f'y_pred: {y_pred}')
            # print(f'y_pred.squeeze: {y_pred.squeeze()}')

            # Compute Loss
            loss = criterion(y_pred.squeeze(), targets)
            loss_values.append(loss.item())
            print(f'Epoch {epoch} train loss: {loss.item()}')

            # Backward pass
            loss.backward()
            optimizer.step()

    return model, loss_values


def test_model(model, data_loader, device):
    model.eval()
    y_pred = []
    y_test = []
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        y_pred.append(model(data))
        y_test.append(targets)
    y_pred = torch.stack(y_pred).squeeze()
    y_test = torch.stack(y_test).squeeze()
    y_pred = y_pred.argmax(dim=1, keepdim=True).squeeze()
    print(classification_report(y_test.cpu(), y_pred.cpu()))


def split(data):
    train_tmp, test = train_test_split(data, test_size=0.2, random_state=SEED)
    train, val = train_test_split(train_tmp, test_size=0.1, random_state=SEED)

    return train, test, val


def run_mlp(df: pd.DataFrame):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dataset = MovieDataset(df)

    hidden_size = 64  # TODO: review?
    num_epochs = 100
    batch = 32
    learning_rate = 0.001

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=42)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch, shuffle=True)

    model = Feedforward(dataset.X.shape[1], hidden_size)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model, loss_values = train_model(model, criterion, optimizer, num_epochs, train_loader, device)
    test_model(model, val_loader, device)

    plt.plot(loss_values)
    plt.title(f'Number of epochs: {num_epochs}')
    plt.show()
