import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset, DataLoader

from src.data.dataset import MovieDataset
from src.utils.const import SEED
import sklearn.metrics as sm


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.hid1 = nn.Linear(input_size, hidden_size)
        self.hid2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 10)
        dropout = 0.4
        self.input_size = input_size
        self.model = nn.Sequential(
            self.hid1,
            nn.ReLU(),
            nn.Dropout(dropout),
            self.hid2,
            nn.ReLU(),
            self.output,
        )
        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

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
    print(f'{len(y_test)} - {len(y_pred)}')
    y_pred = torch.stack(y_pred).squeeze()
    y_test = torch.stack(y_test).squeeze()
    y_pred = y_pred.argmax(dim=1, keepdim=True).squeeze()
    # print(f'{type(y_pred)},{y_test}')
    # mean_squared_error(np.array(y_test), np.array(y_pred))
    # print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))
    score = torch.sum((y_pred.squeeze() == y_test).float()) / y_test.shape[0]
    print('Test score', score.numpy())


def accuracy(model, data_loader, device):
    # assumes model.eval()
    # granular but slow approach
    n_correct = 0
    n_wrong = 0
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            oupt = model(data)  # logits form

        big_idx = torch.argmax(oupt)  # [0] [1] or [2]
        print(f'big: {big_idx} == {targets}')
        if big_idx == targets:
            n_correct += 1
        else:
            n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc


def split(data):
    train_tmp, test = train_test_split(data, test_size=0.2, random_state=SEED)
    train, val = train_test_split(train_tmp, test_size=0.1, random_state=SEED)

    return train, test, val


def run_mlp(df: pd.DataFrame):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dataset = MovieDataset(df)

    # X_train, X_test, X_val = split(dataset.X)
    # y_train, y_test, y_val = split(dataset.y)

    hidden_size = 64
    num_epochs = 10
    learning_rate = 0.001
    batch = 16

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=42)
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=True)

    # y_train = train_dataset.y[train_subset.indices].numpy()
    # y_val = train_dataset.y[val_subset.indices].numpy()
    # plt.subplot(1, 2, 1)
    # plt.hist(y_train, bins=len(np.unique(y_train)), histtype='bar', ec='black')
    # plt.subplot(1, 2, 2)
    # plt.hist(y_val, bins=len(np.unique(y_val)), histtype='bar', ec='black')
    # plt.show()

    model = Feedforward(dataset.X.shape[1], hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # test_model(model, val_loader, device)
    # test_model(model, val_loader, device)
    model, loss_values = train_model(model, criterion, optimizer, num_epochs, train_loader, device)
    print(accuracy(model, val_loader, device))
    plt.plot(loss_values)
    plt.title("Number of epochs: {}".format(num_epochs))
    plt.show()
