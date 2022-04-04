import os

import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

from preprocessing import preprocessing
from settings import MOVIE_LENS_URL, DATASETS_DIR, TMDB_API_URL
from utility import retrieve_csv, request_features_tmdb


def overview(dir_path: str):
    for dataset in os.listdir(dir_path):
        ext = os.path.splitext(dataset)[-1].lower()
        filepath = os.path.join(dir_path, dataset)
        if ext == '.csv':
            df = pd.read_csv(filepath, encoding='utf-8')
            print(f'Filename: {os.path.basename(filepath)}')
            print(f'Shape: {df.shape}')
            print(f'Columns: {df.columns}', end='\n\n')


def main() -> int:
    if not os.path.exists(DATASETS_DIR):
        os.mkdir(DATASETS_DIR)

    if not retrieve_csv(MOVIE_LENS_URL, DATASETS_DIR):
        return 1

        # overview(DATASETS_DIR)
    preprocessing()

    return 0


if __name__ == '__main__':
    exit(main())
