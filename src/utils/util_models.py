import random
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import utils


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


def balancer(train_target: np.ndarray) -> utils.data.WeightedRandomSampler:
    counts = np.bincount(train_target)
    if counts.any(0):
        np.seterr(divide='ignore')
        labels_weights = 1. / counts
        labels_weights[np.isinf(labels_weights)] = 0
    else:
        np.seterr(divide=None)
        labels_weights = 1. / counts
    weights = torch.tensor(labels_weights[train_target], dtype=torch.float)
    return utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)


def add_row_to_df(cfg: int, fold: int, df: pd.DataFrame, loss_test: float, acc_test: float,
                  f1_test: float,
                  list_fold_stat: List) -> pd.DataFrame:

    row_stat = {
        'cfg': cfg,
        'fold': fold,
        'loss_test': loss_test,
        'acc_test': acc_test,
        'f1_test': f1_test
    }

    for metric in list_fold_stat[0].keys():
        numbers = [stat_fold[metric] for stat_fold in list_fold_stat]
        row_stat[f'mean_{metric}'] = [np.mean(numbers)]
        row_stat[f'std_{metric}'] = [np.std(numbers)]

    df = pd.concat([df, pd.DataFrame(data=row_stat)], ignore_index=True)
    return df


def get_set_params(prod, num_sets: int, selected_set: int):
    if selected_set > num_sets:
        selected_set = num_sets - 1
    elif selected_set <= -1:
        return prod

    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]

    tagged_list = [(index,) + element for index, element in enumerate(list(prod))]
    return chunkify(tagged_list, num_sets)[selected_set]
