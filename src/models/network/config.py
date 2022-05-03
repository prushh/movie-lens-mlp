import numpy as np
import torch.nn as nn

parameters = {
    'input_act': [nn.ReLU(), nn.LeakyReLU(), nn.Tanh()],
    'hidden_act': [nn.ReLU(), nn.LeakyReLU()],
    'dropout': np.arange(0, 1.1, .1),
    'batch_norm': [False, True],
    'output_fn': [None, nn.Softmax()]
}
