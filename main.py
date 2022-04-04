import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from settings import MOVIE_LENS_URL, DATASETS_DIR, YEAR_ENCODING, folders_name, external_tmdb_csv_names, \
    raw_csv_names, IMDB_URL, external_tsv_names
from utility import retrieve_tmdb, retrieve_movie_lens, retrieve_imdb


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

    for folder in folders_name:
        folder_path = os.path.join(DATASETS_DIR, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    raw_path = os.path.join(DATASETS_DIR, 'raw')
    if not retrieve_movie_lens(MOVIE_LENS_URL, raw_path, raw_csv_names):
        return 1

    movies = pd.read_csv(os.path.join(raw_path, 'movies.csv'), encoding='utf-8')
    links = pd.read_csv(os.path.join(raw_path, 'links.csv'), encoding='utf-8')
    tags = pd.read_csv(os.path.join(raw_path, 'tags.csv'), encoding='utf-8')
    ratings = pd.read_csv(os.path.join(raw_path, 'ratings.csv'), encoding='utf-8')

    external_path = os.path.join(DATASETS_DIR, 'external')
    features = {'budget', 'revenue', 'adult'}
    if not retrieve_tmdb(links, external_path, external_tmdb_csv_names, features):
        return 1

    if not retrieve_imdb(IMDB_URL, external_path, external_tsv_names):
        return 1

    # overview(DATASETS_DIR)
    preprocessing()

    return 0


if __name__ == '__main__':
    exit(main())
