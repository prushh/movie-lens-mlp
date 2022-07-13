from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

from src.utils.const import NUM_BINS


class MovieDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.idx_column = {}
        for idx, col_name in enumerate(df.columns):
            self.idx_column[col_name] = idx

        X, y_continuous = self.data_target_split(df)

        self.num_classes = NUM_BINS
        y = self._discretize(y_continuous)

        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        return self.X[idx, :], self.y[idx]

    @staticmethod
    def data_target_split(df: pd.DataFrame) -> Tuple:
        y = df['rating_mean']
        X = df.drop(columns='rating_mean').to_numpy()
        return X, y

    def _discretize(self, target: pd.Series) -> pd.Series:
        y = pd.cut(target, bins=self.num_classes, labels=False)
        return y

    def scale(self, train_idx, test_idx, scaler, features: List[int]):
        train_data = self.X[train_idx]
        test_data = self.X[test_idx]

        for feature in features:
            feature_train = train_data[:, feature].reshape(-1, 1)
            feature_test = test_data[:, feature].reshape(-1, 1)

            scaled_train = np.squeeze(scaler.fit_transform(feature_train))
            scaled_test = np.squeeze(scaler.transform(feature_test))

            self.X[train_idx, feature] = torch.tensor(scaled_train, dtype=torch.float)
            self.X[test_idx, feature] = torch.tensor(scaled_test, dtype=torch.float)

    def normalize(self, train_idx, test_idx, norm: str = 'l2'):
        train_data = self.X[train_idx]
        test_data = self.X[test_idx]

        no_cat_columns = list(set(range(train_data.shape[1])) - set(range(2, 21)))

        cat_train_data = train_data[:, list(range(2, 21))]
        cat_test_data = test_data[:, list(range(2, 21))]
        no_cat_train_data = train_data[:, no_cat_columns]
        no_cat_test_data = test_data[:, no_cat_columns]

        norm_train = normalize(no_cat_train_data, norm=norm)
        norm_test = normalize(no_cat_test_data, norm=norm)

        all_train = np.concatenate((norm_train, cat_train_data), axis=1)
        all_test = np.concatenate((norm_test, cat_test_data), axis=1)

        self.X[train_idx, :] = torch.tensor(all_train, dtype=torch.float)
        self.X[test_idx, :] = torch.tensor(all_test, dtype=torch.float)
