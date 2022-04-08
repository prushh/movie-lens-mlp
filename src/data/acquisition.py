import os
from typing import List, Set

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.const import DATA_DIR, DATA_SUB_DIRS, RAW_CSV_NAMES, MOVIE_LENS_URL, IMDB_URL, EXTERNAL_IMDB_CSV_NAMES, \
    EXTERNAL_TMDB_CSV_NAMES, TMDB_API_URL
from src.utils.util import missing_files, download_file, unzip, output_datasets, get_missing_files, gunzip, tsv_to_csv, \
    csv_to_tsv_gz_ext, request_features_tmdb


def retrieve_tmdb(df: pd.DataFrame, out_dir: str, dir_csv_names: List, features: Set[str],
                  filename: str = 'tmdb-features-old.csv', log: bool = False) -> bool:
    """
    If not exists create a .csv that contains new features, retrieved by API requests to TMDB service
    :param df: the DataFrame that contains movies_id and tmdb_id
    :param out_dir: the folder in which to save the csv files
    :param dir_csv_names: the list of all files to check the existence
    :param features: the list with the features to extract
    :param filename: the name of the new .csv file
    :param log: if True print single API request and DataFrame.shape[0] information
    :return: False if something went wrong during the requests, True instead
    """
    to_download = missing_files(out_dir, dir_csv_names)

    if to_download:
        load_dotenv()
        token = os.environ.get('TMDB_API_KEY')

        url = TMDB_API_URL.substitute(tmdb_id='', api_key='')
        print(f'Some external TMDB data missing, start API requests from{os.path.dirname(url)}')

        tmdb_features = pd.DataFrame()
        for (movie_id, _, tmdb_id) in df.itertuples(index=False):
            url = TMDB_API_URL.substitute(tmdb_id=tmdb_id, api_key=token)
            try:
                sample = request_features_tmdb(
                    url,
                    movie_id,
                    tmdb_id,
                    features
                )
            except requests.exceptions.RequestException:
                if log:
                    print('Save after error.')
                    tmdb_features.to_csv('data/external/tmdb-features-tmp.csv', encoding='utf-8', index=False)
                print('Error, something went wrong during the API requests.')
                return False
            else:
                tmdb_features = pd.concat([tmdb_features, sample], ignore_index=True)
                if log:
                    print(f'Samples: {tmdb_features.shape[0]}')

        tmdb_features_path = os.path.join(out_dir, filename)
        tmdb_features.to_csv(tmdb_features_path, index=False, encoding='utf-8')
        output_datasets([filename])
    else:
        print('No needed to download external TMDB data.')
        print('-' * 30)

    return True


def retrieve_imdb(url: str, out_dir: str, dir_csv_names: List[str]) -> bool:
    """
    Retrieve, if necessary, IMDB data specified in the dir_csv_names list
    :param url: the web URI to the resource
    :param out_dir: the directory where the files are saved
    :param dir_csv_names: the list with all the files to check
    :return: False if something went wrong during the download or True instead
    """
    missing_csv_names = get_missing_files(out_dir, dir_csv_names)

    if missing_csv_names:
        print(f'Some external IMDB data missing, start download from {os.path.dirname(url)}')
        extracted = []
        missing_tsv_names = csv_to_tsv_gz_ext(missing_csv_names)
        for filename_csv, filename_tsv_gz in zip(missing_csv_names, missing_tsv_names):
            file_url = os.path.join(url, filename_tsv_gz)
            filepath_tsv = os.path.join(out_dir, filename_tsv_gz)
            if not os.path.exists(filepath_tsv):
                if not download_file(file_url, filepath_tsv, filename_tsv_gz):
                    return False

            filepath_csv = os.path.join(out_dir, filename_csv)
            gunzip(filepath_tsv, filepath_csv)
            converted = tsv_to_csv(filepath_csv)
            extracted.append(converted)

            os.remove(filepath_tsv)

        output_datasets(extracted)
    else:
        print('No needed to download external IMDB data.')
        print('-' * 30)

    return True


def retrieve_movie_lens(url: str, out_dir: str, dir_csv_names: List[str]) -> bool:
    """
    Retrieve, if necessary, MovieLens data specified in the dir_csv_names list
    :param url: the web URI to the resource
    :param out_dir: the directory where the files are saved
    :param dir_csv_names: the list with all the files to check
    :return: False if something went wrong during the download or True instead
    """
    to_download = missing_files(out_dir, dir_csv_names)

    if to_download:
        # At least one csv file missing, download zip
        filename = os.path.basename(url)
        filepath = os.path.join(out_dir, filename)
        if not os.path.exists(filepath):
            print(f'Some raw MovieLens data missing, start download from {os.path.dirname(url)}')
            if not download_file(url, filepath, filename):
                return False

        extracted = unzip(filepath, out_dir)
        os.remove(filepath)
        output_datasets(extracted)
    else:
        print('No needed to download raw MovieLens data.')
        print('-' * 30)

    return True


def retrieve_datasets() -> bool:
    # Create data dir if not exists
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    # Create external, interim, processed and raw dirs if not exist
    for _dir in DATA_SUB_DIRS:
        dir_path = os.path.join(DATA_DIR, _dir)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    # Retrieve MovieLens datasets
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    if not retrieve_movie_lens(MOVIE_LENS_URL, RAW_DIR, RAW_CSV_NAMES):
        return False

    # Retrieve IMDB dataset
    EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
    if not retrieve_imdb(IMDB_URL, EXTERNAL_DIR, EXTERNAL_IMDB_CSV_NAMES):
        return False

    # Retrieve TMDB dataset by API requests, links DataFrame is needed for tmdbId feature
    links = pd.read_csv(os.path.join(RAW_DIR, 'links.csv'), encoding='utf-8')
    # Specify features to retrieve from https://developers.themoviedb.org/3/movies/get-movie-details
    features = {'imdb_id', 'budget', 'revenue', 'adult'}
    if not retrieve_tmdb(links, EXTERNAL_DIR, EXTERNAL_TMDB_CSV_NAMES, features, log=True):
        return False

    return True
