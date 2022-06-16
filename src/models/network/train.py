from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils import tensorboard

from src.utils.util_models import get_correct_samples


def train(writer: tensorboard.SummaryWriter,
          model: nn.Module,
          train_loader: utils.data.DataLoader,
          device: torch.device,
          optimizer: optim,
          criterion: Callable[[torch.Tensor, torch.Tensor], float],
          log_interval: int,
          epoch: int) -> Tuple:
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

    y_pred = []
    y_test = []

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

        class_prob = [F.softmax(elm, dim=0) for elm in scores]

        y_pred.append(class_prob)
        y_test.append(targets)

        if log_interval > 0:
            if idx_batch % log_interval == 0:
                running_loss = loss_train / samples_train
                global_step = idx_batch + (epoch * num_batches)
                writer.add_scalar('Metrics/Loss_Train_IT', running_loss, global_step)
                # # Visualize images on tensorboard
                # indices_random = torch.randperm(images.size(0))[:4]
                # writer.add_images('Samples/Train', denormalize(images[indices_random]), global_step)

    y_pred_prob = torch.cat([torch.stack(batch) for batch in y_pred])
    y_test = torch.cat(y_test)
    y_pred = y_pred_prob.argmax(dim=1, keepdim=True)

    loss_train /= samples_train
    accuracy_training = 100. * correct / samples_train
    f_score = f1_score(y_test.cpu(), y_pred.cpu(), average='weighted')

    # print(loss_train, type(loss_train))
    # print(accuracy_training, type(accuracy_training))
    # print(f_score, type(f_score))
    # exit(1)

    return loss_train, accuracy_training, f_score
