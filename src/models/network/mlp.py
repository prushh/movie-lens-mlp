import itertools
import os
from typing import Dict, Union

from timeit import default_timer as timer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils as utils
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch.nn import CrossEntropyLoss
from torchinfo import summary
from torch.utils import tensorboard
from torch.optim import lr_scheduler

from src.data.dataset import MovieDataset
from src.models.config import param_layers, param_grid_mlp
from src.models.network.train import train
from src.models.network.validate import validate
from src.utils.const import NETWORK_RESULTS_DIR, NETWORK_RESULT_CSV
from src.utils.util_models import balancer, add_row_to_df


class MovieNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            input_act: nn.Module,
            hidden_size: int,
            hidden_act: nn.Module,
            num_hidden_layers: int,
            output_fn,
            num_classes: int,
            dropout: float = 0.0,
            batch_norm: bool = False
    ) -> None:
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


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('\t\tINFO: Early stopping')
                self.early_stop = True


def training_loop(
        writer: tensorboard.SummaryWriter,
        num_epochs: int,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        log_interval: int,
        model: nn.Module,
        loader_train: utils.data.DataLoader,
        loader_val: utils.data.DataLoader,
        device: torch.device,
        verbose: bool = True
) -> Dict:
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
    early_stopping = EarlyStopping()
    loop_start = timer()

    losses_values = []
    train_acc_values = []
    train_f1_values = []
    val_f1_values = []
    val_acc_values = []
    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        loss_train, accuracy_train, f_score_train = train(writer, model, loader_train, device,
                                                          optimizer, criterion, log_interval,
                                                          epoch)

        loss_val, accuracy_val, f_score_val = validate(model, loader_val, device, criterion)
        time_end = timer()

        losses_values.append(loss_train)
        train_acc_values.append(accuracy_train)
        val_acc_values.append(accuracy_val)
        train_f1_values.append(f_score_train)
        val_f1_values.append(f_score_val)

        lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' Loss: Train = [{loss_train:.4f}] - Val = [{loss_val:.4f}] '
                  f' Accuracy: Train = [{accuracy_train:.2f}%] - Val = [{accuracy_val:.2f}%] '
                  f' F1: Train = [{f_score_train:.3f}] - Val = [{f_score_val:.3f}] '
                  f' Time one epoch (s): {(time_end - time_start):.4f} ')

        # Plot to tensorboard
        writer.add_scalar('Hyper parameters/Learning Rate', lr, epoch)
        writer.add_scalars('Metrics/Losses', {'Train': loss_train, 'Val': loss_val}, epoch)
        writer.add_scalars('Metrics/Accuracy', {'Train': accuracy_train, 'Val': accuracy_val}, epoch)
        writer.flush()

        # Increases the internal counter
        if scheduler:
            scheduler.step()

        early_stopping(loss_val)
        if early_stopping.early_stop:
            break

    loop_end = timer()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {time_loop :.3f}')

    return {'loss_values': losses_values,
            'train_acc_values': train_acc_values,
            'val_acc_values': val_acc_values,
            'train_f1_values': train_f1_values,
            'val_f1_values': val_f1_values,
            'time': time_loop}


def execute(
        name_train: str,
        network: nn.Module,
        optimizer: Union[torch.optim.Adam, torch.optim.SGD],
        num_epochs: int,
        data_loader_train: torch.utils.data.DataLoader,
        data_loader_val: torch.utils.data.DataLoader,
        device: torch.device
) -> Dict:
    """
    Executes the training loop
    :param name_train: the name for the log sub-folder
    :param network: the network to train
    :param optimizer: the optimizer
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
    # optimizer = optim.Adam(network.parameters(), lr=starting_lr, weight_decay=0.000001)

    # Learning Rate schedule: decays the learning rate by a factor of `gamma`
    # every `step_size` epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    statistics = training_loop(writer, num_epochs, optimizer, scheduler,
                               log_interval, network, data_loader_train,
                               data_loader_val, device, verbose=True)
    writer.close()

    best_epoch = np.argmax(statistics['val_acc_values']) + 1
    best_accuracy = statistics['val_acc_values'][best_epoch - 1]

    print(f'Best val accuracy: {best_accuracy:.2f} epoch: {best_epoch}.')
    # return statistics['val_acc_values'][-1]
    return {'loss': np.mean(statistics['loss_values']),
            'acc_val': np.mean(statistics['val_acc_values']),
            'acc_train': np.mean(statistics['train_acc_values']),
            'f1_train': np.mean(statistics['train_f1_values']),
            'f1_val': np.mean(statistics['val_f1_values'])}


def mlp(df: pd.DataFrame):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('Using device:', torch.cuda.get_device_name(device))

    dataset = MovieDataset(df)

    features = [
        dataset.idx_column['year'],
        dataset.idx_column['title_length'],
        dataset.idx_column['tag_count'],
        dataset.idx_column['runtime'],
        dataset.idx_column['rating_count']
    ]

    n_splits = 5
    cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True)
    df = pd.DataFrame()

    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(dataset.X, y=dataset.y), 1):
        hyper_parameters_model = itertools.product(
            param_layers['input_act'],
            param_layers['hidden_act'],
            param_layers['hidden_size'],
            param_layers['num_hidden_layers'],
            param_layers['dropout'],
            param_layers['batch_norm'],
            param_layers['output_fn'],
            param_grid_mlp['starting_lr'],
            param_grid_mlp['num_epochs'],
            param_grid_mlp['batch_size'],
            param_grid_mlp['optim'],
            param_grid_mlp['momentum'],
            param_grid_mlp['weight_decay'],
        )

        print('=' * 65)
        print(f'Fold {fold}')

        list_fold_stat = []
        best_cfg_network = None
        max_f1_test = 0
        data_test = utils.data.Subset(dataset, test_idx)

        num_workers = 2
        loader_test = utils.data.DataLoader(data_test, batch_size=1,
                                            shuffle=False,
                                            num_workers=num_workers)

        for idx, (input_act,
                  hidden_act,
                  hidden_size,
                  num_hidden_layers,
                  dropout,
                  batch_norm,
                  _,
                  starting_lr,
                  num_epochs,
                  batch_size,
                  optimizer_class,
                  momentum,
                  weight_decay) in enumerate(hyper_parameters_model):

            best_val_network = None
            max_f1_val = 0

            cfg = (input_act, hidden_act, hidden_size, num_hidden_layers, dropout, batch_norm, starting_lr, num_epochs,
                   batch_size, optimizer_class, momentum, weight_decay)

            cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True)

            for inner_fold, (inner_train_idx, val_idx) in enumerate(
                    cv_inner.split(dataset.X[train_idx], y=dataset.y[train_idx]), 1):

                # Balancing
                train_target = dataset.y[inner_train_idx]
                sampler = balancer(train_target)

                # Scaling and normalization
                scaler = preprocessing.MinMaxScaler()
                dataset.scale(train_idx, test_idx, scaler, features)
                dataset.normalize(train_idx, test_idx)

                data_train = utils.data.Subset(dataset, inner_train_idx)
                data_val = utils.data.Subset(dataset, val_idx)

                loader_train = utils.data.DataLoader(data_train, batch_size=batch_size,
                                                     sampler=sampler,
                                                     pin_memory=True,
                                                     num_workers=num_workers)

                loader_val = utils.data.DataLoader(data_val, batch_size=1,
                                                   shuffle=False,
                                                   num_workers=num_workers)

                input_size = dataset.X.shape[1]
                num_classes = dataset.num_classes
                network = MovieNet(input_size=input_size,
                                   input_act=input_act,
                                   hidden_size=hidden_size,
                                   hidden_act=hidden_act,
                                   num_hidden_layers=num_hidden_layers,
                                   dropout=dropout,
                                   output_fn=None,
                                   num_classes=num_classes)
                network.reset_weights()
                network.to(device)

                if fold == 1 and inner_fold == 1:
                    print('=' * 65)
                    print(f'Configuration [{idx}]: {cfg}')
                    summary(network)

                # TODO: fix experiment name
                name_train = f'movie_net_experiment_{idx}'

                if optimizer_class == torch.optim.Adam:
                    optimizer = optimizer_class(network.parameters(),
                                                lr=starting_lr,
                                                weight_decay=weight_decay)
                else:
                    optimizer = optimizer_class(network.parameters(),
                                                lr=starting_lr,
                                                momentum=momentum,
                                                weight_decay=weight_decay)

                fold_stat = execute(name_train,
                                    network,
                                    optimizer,
                                    num_epochs,
                                    loader_train,
                                    loader_val,
                                    device)
                list_fold_stat.append(fold_stat)

                if fold_stat['f1_val'] >= max_f1_val:
                    max_f1_val = fold_stat['f1_val']
                    best_val_network = network

            criterion = CrossEntropyLoss()
            loss_test, acc_test, f1_test = validate(best_val_network, loader_test, device, criterion)
            print(f'Test {fold}, loss={loss_test:3f}, accuracy={acc_test:3f}, f1={f1_test:3f}')

            df = add_row_to_df(idx, fold, df, loss_test, acc_test, f1_test, list_fold_stat)
            df.to_csv(os.path.join(NETWORK_RESULT_CSV, 'out.csv'), encoding='utf-8')

            # Find the best cfg network between the already computed configurations
            if f1_test >= max_f1_test:
                max_f1_test = f1_test
                best_cfg_network = best_val_network

        if not os.path.exists(NETWORK_RESULTS_DIR):
            os.mkdir(NETWORK_RESULTS_DIR)
        path = os.path.join(NETWORK_RESULTS_DIR, f'best_network_{fold}.pt')
        torch.save(best_cfg_network.state_dict(), path)

    df.to_csv(os.path.join(NETWORK_RESULT_CSV, 'out.csv'), encoding='utf-8')
