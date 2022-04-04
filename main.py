import os

import pandas as pd

from preprocessing import preprocessing
from settings import MOVIE_LENS_URL, DATASETS_DIR, folders_name, external_tmdb_csv_names, raw_csv_names, IMDB_URL, \
    external_tsv_names
from utility import retrieve_tmdb, retrieve_movie_lens, retrieve_imdb


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

    # TODO: import here of all raw datasets
    links = pd.read_csv(os.path.join(raw_path, 'links.csv'), encoding='utf-8')

    external_path = os.path.join(DATASETS_DIR, 'external')
    features = {'budget', 'revenue', 'adult'}
    if not retrieve_tmdb(links, external_path, external_tmdb_csv_names, features):
        return 1

    if not retrieve_imdb(IMDB_URL, external_path, external_tsv_names):
        return 1

    # TODO: check if interim and processed .csv file exist
    preprocessing()

    return 0


if __name__ == '__main__':
    exit(main())
