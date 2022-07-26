import os
from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score

from src.utils.const import NETWORK_RESULTS_DIR
from src.utils.util_models import get_correct_samples
from src.visualization.visualize import plot_roc


def validate(model: nn.Module,
             data_loader: utils.data.DataLoader,
             device: torch.device,
             criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> Tuple:
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

    y_pred = []
    y_test = []

    model = model.eval()
    with torch.no_grad():
        for idx_batch, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            scores = model(data)

            loss = criterion(scores, targets)
            loss_val += loss.item() * len(data)
            samples_val += len(data)
            correct += get_correct_samples(scores, targets)

            class_prob = [F.softmax(elm, dim=0) for elm in scores]

            y_pred.append(class_prob)
            y_test.append(targets)

    y_pred_prob = torch.cat([torch.stack(batch) for batch in y_pred])
    y_test = torch.cat(y_test)
    y_pred = y_pred_prob.argmax(dim=1, keepdim=True)

    loss_val /= samples_val
    accuracy = 100. * correct / samples_val
    f_score = f1_score(y_test.cpu(), y_pred.cpu(), average='weighted')

    return loss_val, accuracy, f_score


def test_eval(model_fold: int,
              data_loader: utils.data.DataLoader,
              device: torch.device,
              criterion: Callable[[torch.Tensor, torch.Tensor], float],
              roc: bool) -> Tuple:
    """
    Evaluates the model
    :param model_fold: the saved model specified by fold index
    :param data_loader: the data loader containing the validation or test data
    :param device: the device to use to evaluate the model
    :param criterion: the loss function
    :param roc: the flag to plot roc graph
    :return: the loss value and the accuracy on the validation data
    """
    filename = f'{model_fold}_network.pt'
    filepath = os.path.join(NETWORK_RESULTS_DIR, 'mlp', filename)
    model = torch.load(filepath)

    correct = 0
    samples_val = 0
    loss_val = 0.

    model = model.eval()
    with torch.no_grad():
        y_pred = []
        y_test = []
        for idx_batch, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            scores = model(data)

            loss = criterion(scores, targets)
            loss_val += loss.item() * len(data)
            samples_val += len(data)
            correct += get_correct_samples(scores, targets)

            class_prob = [F.softmax(elm, dim=0) for elm in scores]

            y_pred.append(class_prob)
            y_test.append(targets)

        y_pred_prob = torch.cat([torch.stack(batch) for batch in y_pred])
        y_test = torch.cat(y_test)
        y_pred = y_pred_prob.argmax(dim=1, keepdim=True)

    loss_val /= samples_val
    accuracy = 100. * correct / samples_val
    f_score = f1_score(y_test.cpu(), y_pred.cpu(), average='weighted')

    targets_name = [str(i) for i in np.arange(0, y_pred_prob.shape[1])]

    print(classification_report(y_test, y_pred, target_names=targets_name, zero_division=0))
    plot_roc(y_test=y_test, y_pred_proba=y_pred_prob, model_name='MLP')

    return loss_val, accuracy, f_score
