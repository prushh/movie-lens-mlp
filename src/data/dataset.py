import pandas as pd
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        # return self.X.shape[0]
        pass

    def __getitem__(self, idx: int):
        # return self.X[idx, :], self.y[idx]
        pass
