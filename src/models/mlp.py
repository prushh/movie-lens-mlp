import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

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
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, criterion, optimizer, epochs, data_loader, device):
    model.train()
    loss_values = []
    for epoch in range(epochs):
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(data)

            # Compute Loss
            loss = criterion(y_pred.squeeze(), targets)
            loss_values.append(loss.item())
            print('Epoch {} train loss: {}'.format(epoch, loss.item()))

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


def split(data):
    train_tmp, test = train_test_split(data, test_size=0.2, random_state=SEED)
    train, val = train_test_split(train_tmp, test_size=0.1, random_state=SEED)

    return train, test, val


def run_mlp(df: pd.DataFrame):
    dataset = MovieDataset(df)

    X_train, X_test, X_val = split(dataset.X)
    y_train, y_test, y_val = split(dataset.y)


