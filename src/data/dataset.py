from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        target = 'rating_mean'

        X = df.loc[:, df.columns != target]
        y = df[target]

        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        return self.X[idx, :], self.y[idx]