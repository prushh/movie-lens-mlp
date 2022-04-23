from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scale_method: str = '', norm: str = ''):
        self.df = df
        self._process()

        stratify = df['discrete_mean']
        df_train_tmp, df_test = train_test_split(df, test_size=0.2, stratify=stratify)
        df_train, df_val = train_test_split(df, test_size=0.1, stratify=stratify)
        # TODO: review drop parameter, refactor in a better way
        df_train.reset_index(inplace=True)
        df_test.reset_index(inplace=True)
        df_val.reset_index(inplace=True)

        self.X, self.y_discrete, self.y_continuous = _split_data(df)

        self.X_train, self.y_train, _ = _split_data(df_train)
        self.X_test, self.y_test, _ = _split_data(df_test)
        self.X_val, self.y_val, _ = _split_data(df_val)

        if scale_method != '':
            self._scaling(scale_method)
            print(f'Scaling done using {scale_method}.')

        if norm != '':
            self._normalize(norm)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        return self.X[idx, :], self.y_discrete[idx], self.y_continuous[idx]

    def _process(self, bins: int = 10):
        self.df['discrete_mean'] = pd.cut(self.df['rating_mean'], bins=bins, labels=False)

    def _scaling(self, kind: str):
        scaler = None

        if kind == 'min-max':
            scaler = MinMaxScaler()
        elif kind == 'standardization':
            scaler = StandardScaler()

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.X_val = scaler.transform(self.X_val)

    def _normalize(self, norm: str):
        self.X_train = normalize(self.X_train, norm=norm)
        self.X_test = normalize(self.X_test, norm=norm)
        self.X_val = normalize(self.X_val, norm=norm)


def _split_data(df: pd.DataFrame) -> Tuple:
    continuous_mean = 'rating_mean'
    discrete_mean = 'discrete_mean'

    no_targets = (df.columns != continuous_mean) & (df.columns != discrete_mean)
    X = df.loc[:, no_targets]
    y_discrete = df[discrete_mean]
    y_continuous = df[continuous_mean]

    return (
        torch.FloatTensor(np.array(X)),
        torch.IntTensor(y_discrete),
        torch.FloatTensor(y_continuous)
    )
