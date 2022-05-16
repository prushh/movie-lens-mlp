from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.utils as utils

from src.utils.util_models import get_correct_samples


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
