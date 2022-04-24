import os
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import Subset, DataLoader

from src.data.dataset import MovieDataset
from src.utils.const import SEED, LOG_DIR


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()

        self.hid1 = nn.Linear(input_size, hidden_size)
        self.hid2 = nn.Linear(hidden_size, 512)
        self.hid3 = nn.Linear(512, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        dropout = 0.4
        self.input_size = input_size
        self.model = nn.Sequential(
            self.hid1,
            nn.ReLU(),
            nn.Dropout(dropout),
            self.hid2,
            nn.ReLU(),
            self.hid3,
            nn.ReLU(),
            self.output,
        )
        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.hid3.weight)
        nn.init.zeros_(self.hid3.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.model(x)


def train_model(model, criterion, optimizer, start_epoch, epochs, data_loader, device):
    model.train()
    loss_values = []
    correct = 0
    for epoch in range(start_epoch, epochs):
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
            # correct += (y_pred == targets).float().sum()
            if epoch % 100 == 0:
                dt = time.strftime("%Y_%m_%d-%H_%M_%S")

                fn = "src\\models\\log\\" + str(dt) + str("-") + \
                     str(epoch) + "_checkpoint.pt"

                info_dict = {
                    'epoch': epoch,
                    'net_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(info_dict, fn)

            # accuracy = 100 * correct / len(data_loader)
            # trainset, not train_loader
            # probably x in your case
            # print("Accuracy = {}".format(accuracy))

    return model, loss_values


def restore_train(filepath, model, optimizer):
    chkpt = torch.load(filepath)
    model.load_state_dict(chkpt['net_state'])
    optimizer.load_state_dict(chkpt['optimizer_state'])
    epoch_saved = chkpt['epoch'] + 1
    return epoch_saved


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
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    dataset = MovieDataset(df)

    input_size = dataset.X.shape[1]
    hidden_size = 64
    output_size = dataset.num_classes
    num_epochs = 100
    learning_rate = 0.001
    batch = 128

    # Variable to be used to restore
    restore = False
    filename = '2022_04_16-11_12_51-0_checkpoint.pt'
    # init of epoch start if not resuming
    epoch_start = 0

    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=dataset.y)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=dataset.y[train_idx])

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    features = [
        dataset.map_columns['year'],
        dataset.map_columns['title_length'],
        dataset.map_columns['runtime'],
        dataset.map_columns['rating_count']
    ]
    dataset.scale(train_idx, test_idx, val_idx, scaler, features)
    # dataset.normalize(train_idx)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=True)

    model = Feedforward(input_size, hidden_size, output_size)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if restore:
        filepath = os.path.join(LOG_DIR, filename)
        epoch_start = restore_train(filepath, model, optimizer)
    model, loss_values = train_model(model, criterion, optimizer, epoch_start, num_epochs, train_loader, device)
    print(accuracy(model, val_loader, device))
    plt.plot(loss_values)
    plt.title("Number of epochs: {}".format(num_epochs))
    plt.show()
