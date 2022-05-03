import random
import numpy as np
import torch


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
