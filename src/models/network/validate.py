from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.utils.const import NUM_BINS
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


def test(model: nn.Module,
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
        y_pred = []
        y_test = []
        for idx_batch, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            scores = model(data)
            class_probb = [F.softmax(el, dim=0) for el in scores]

            y_pred.append(class_probb)
            y_test.append(targets)

            #loss = criterion(scores, targets)
            #loss_val += loss.item() * len(data)
            #samples_val += len(data)
            #correct += get_correct_samples(scores, targets)
        y_pred_proba = torch.cat([torch.stack(batch) for batch in y_pred])
        y_test = torch.cat(y_test)
        y_pred = y_pred_proba.argmax(dim=1, keepdim=True)

    targets_name = [str(i) for i in np.arange(0, y_pred_proba.shape[1])]
    print(classification_report(y_test, y_pred, target_names=targets_name, zero_division=0))
    #loss_val /= samples_val
    #accuracy = 100. * correct / samples_val
    classes = [i for i in range(y_pred_proba.shape[1])]

    # binarize: [2] -> [0,0,1,0,0,0,0,0,0]
    y_test = label_binarize(y_test, classes=classes)
    y_pred= label_binarize(y_pred, classes=classes)
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NUM_BINS):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(NUM_BINS):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    return (1,1)
