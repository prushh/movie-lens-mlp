import os
import random
from typing import Tuple, Callable, Dict

from timeit import default_timer as timer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torchinfo import summary
from torch.utils import tensorboard
from torch.optim import lr_scheduler

from src.data.dataset import MovieDataset


class MovieNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 input_act: nn.Module,
                 hidden_size: int,
                 hidden_act: nn.Module,
                 num_hidden_layers: int,
                 output_fn,
                 num_classes: int,
                 dropout: float = 0.0,
                 batch_norm: bool = False):
        super(MovieNet, self).__init__()

        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            input_act
        ])

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))

            self.layers.append(hidden_act)

            if dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(hidden_size, num_classes))

        if output_fn:
            self.layers.append(output_fn)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reset_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def fix_random(seed: int) -> None:
    """
    Fix all the possible sources of randomness
    :param seed: the seed to use
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_correct_samples(scores: torch.Tensor, labels: torch.Tensor) -> int:
    """
    Gets the number of correctly classified examples
    :param scores: the scores predicted with the network
    :param labels: the class labels
    :return: the number of correct samples
    """
    classes_predicted = torch.argmax(scores, 1)
    return (classes_predicted == labels).sum().item()


# Train one epoch
def train(writer: tensorboard.SummaryWriter,
          model: nn.Module,
          train_loader: utils.data.DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          criterion: Callable[[torch.Tensor, torch.Tensor], float],
          log_interval: int,
          epoch: int) -> Tuple[float, float]:
    """
    Trains a neural network for one epoch
    :param writer: the summary writer for tensorboard
    :param model: the model to train
    :param train_loader: the data loader containing the training data
    :param device: the device to use to train the model
    :param optimizer: the optimizer to use to train the model
    :param criterion: the loss to optimize
    :param log_interval: the log interval
    :param epoch: the number of the current epoch
    :return: the cross entropy Loss value and the accuracy on the training data.
    """
    correct = 0
    samples_train = 0
    loss_train = 0
    num_batches = len(train_loader)

    model.train()
    for idx_batch, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        scores = model(data)

        loss = criterion(scores, targets)
        loss_train += loss.item() * len(data)
        samples_train += len(data)

        loss.backward()
        optimizer.step()
        correct += get_correct_samples(scores, targets)

        if log_interval > 0:
            if idx_batch % log_interval == 0:
                running_loss = loss_train / samples_train
                global_step = idx_batch + (epoch * num_batches)
                writer.add_scalar('Metrics/Loss_Train_IT', running_loss, global_step)
                # # Visualize images on tensorboard
                # indices_random = torch.randperm(images.size(0))[:4]
                # writer.add_images('Samples/Train', denormalize(images[indices_random]), global_step)

    loss_train /= samples_train
    accuracy_training = 100. * correct / samples_train
    return loss_train, accuracy_training


# Validate one epoch
def validate(model: nn.Module,
             data_loader: utils.data.DataLoader,
             device: torch.device,
             criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> Tuple[float, float]:
    """
    Evaluates the model
    :param model: the model to evaluate
    :param data_loader: the data loader containing the validation or test data
    :param device: the device to use to evaluate the model
    :param criterion: the loss function
    :return: the loss value and the accuracy on the validation data
    """
    correct = 0
    samples_val = 0
    loss_val = 0.
    model = model.eval()
    with torch.no_grad():
        for idx_batch, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            scores = model(data)

            loss = criterion(scores, targets)
            loss_val += loss.item() * len(data)
            samples_val += len(data)
            correct += get_correct_samples(scores, targets)

    loss_val /= samples_val
    accuracy = 100. * correct / samples_val
    return loss_val, accuracy


def training_loop(writer: tensorboard.SummaryWriter,
                  num_epochs: int,
                  optimizer: torch.optim,
                  scheduler: torch.optim.lr_scheduler,
                  log_interval: int,
                  model: nn.Module,
                  loader_train: utils.data.DataLoader,
                  loader_val: utils.data.DataLoader,
                  device: torch.device,
                  verbose: bool = True) -> Dict:
    """
    Executes the training loop
    :param writer: the summary writer for tensorboard
    :param num_epochs: the number of epochs
    :param optimizer: the optimizer to use
    :param scheduler: the scheduler for the learning rate
    :param log_interval: interval to print on tensorboard
    :param model: the mode to train
    :param loader_train: the data loader containing the training data
    :param loader_val: the data loader containing the validation data
    :param device: the device to use to train the model
    :param verbose: if true print the value of loss
    :return: Dict with statistics
    """
    criterion = nn.CrossEntropyLoss()
    loop_start = timer()

    losses_values = []
    train_acc_values = []
    val_acc_values = []
    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        loss_train, accuracy_train = train(writer, model, loader_train, device,
                                           optimizer, criterion, log_interval,
                                           epoch)
        loss_val, accuracy_val = validate(model, loader_val, device, criterion)
        time_end = timer()

        losses_values.append(loss_train)
        train_acc_values.append(accuracy_train)
        val_acc_values.append(accuracy_val)

        lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' Loss: Train = [{loss_train:.4f}] - Val = [{loss_val:.4f}] '
                  f' Accuracy: Train = [{accuracy_train:.2f}%] - Val = [{accuracy_val:.2f}%] '
                  f' Time one epoch (s): {(time_end - time_start):.4f} ')

        # Plot to tensorboard
        writer.add_scalar('Hyper parameters/Learning Rate', lr, epoch)
        writer.add_scalars('Metrics/Losses', {'Train': loss_train, 'Val': loss_val}, epoch)
        writer.add_scalars('Metrics/Accuracy', {'Train': accuracy_train, 'Val': accuracy_val}, epoch)
        writer.flush()

        # Increases the internal counter
        if scheduler:
            scheduler.step()

    loop_end = timer()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {time_loop :.3f}')

    return {'loss_values': losses_values,
            'train_acc_values': train_acc_values,
            'val_acc_values': val_acc_values,
            'time': time_loop}


def execute(name_train: str,
            network: nn.Module,
            starting_lr: float,
            num_epochs: int,
            data_loader_train: torch.utils.data.DataLoader,
            data_loader_val: torch.utils.data.DataLoader,
            device: torch.device) -> None:
    """
    Executes the training loop
    :param name_train: the name for the log sub-folder
    :param network: the network to train
    :param starting_lr: the staring learning rate
    :param num_epochs: the number of epochs
    :param data_loader_train: the data loader with training data
    :param data_loader_val: the data loader with validation data
    :param device: the device to use to train the model
    :return: None
    """
    # Visualization
    log_interval = 20
    network_path = os.path.dirname(__file__)
    log_dir = os.path.join(network_path, 'logs', name_train)
    writer = tensorboard.SummaryWriter(log_dir)

    # Optimization
    optimizer = optim.Adam(network.parameters(), lr=starting_lr)

    # Learning Rate schedule: decays the learning rate by a factor of `gamma`
    # every `step_size` epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    statistics = training_loop(writer, num_epochs, optimizer, scheduler,
                               log_interval, network, data_loader_train,
                               data_loader_val, device)
    writer.close()

    best_epoch = np.argmax(statistics['val_acc_values']) + 1
    best_accuracy = statistics['val_acc_values'][best_epoch - 1]

    print(f'Best val accuracy: {best_accuracy:.2f} epoch: {best_epoch}.')


def run_mlp(df: pd.DataFrame):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    '''
    if device.type == 'cuda':
        print('Using device:', torch.cuda.get_device_name(device))'''

    fix_random(42)
    dataset = MovieDataset(df)
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=dataset.y)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=dataset.y[train_idx])

    scaler = preprocessing.MinMaxScaler()
    features = [
        dataset.map_columns['year'],
        dataset.map_columns['title_length'],
        dataset.map_columns['tag_count'],
        dataset.map_columns['runtime'],
        dataset.map_columns['rating_count']
    ]
    dataset.scale(train_idx, test_idx, val_idx, scaler, features)
    dataset.normalize(train_idx, test_idx, val_idx)

    train_target = dataset.y[train_idx]
    counts = np.bincount(train_target)
    labels_weights = 1. / counts
    weights = torch.tensor(labels_weights[train_target], dtype=torch.float)
    sampler = utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

    data_train = utils.data.Subset(dataset, train_idx)
    data_val = utils.data.Subset(dataset, val_idx)

    num_workers = 2
    size_batch = 64
    loader_train = utils.data.DataLoader(data_train, batch_size=size_batch,
                                         sampler=sampler,
                                         pin_memory=True,
                                         num_workers=num_workers)
    loader_val = utils.data.DataLoader(data_val, batch_size=size_batch,
                                       shuffle=False,
                                       num_workers=num_workers)

    input_size = dataset.X.shape[1]
    hidden_size = 64
    num_classes = dataset.num_classes
    network = MovieNet(input_size=input_size,
                       input_act=nn.LeakyReLU(),
                       hidden_size=hidden_size,
                       hidden_act=nn.LeakyReLU(),
                       num_hidden_layers=3,
                       dropout=0.5,
                       output_fn=None,
                       num_classes=num_classes)
    network.reset_weights()
    network.to(device)
    summary(network)

    name_train = 'movie_net'
    lr = 0.001
    num_epochs = 100
    execute(name_train, network, lr, num_epochs, loader_train, loader_val, device)
