import gzip
import os
import sys
from typing import List
from zipfile import ZipFile

import pandas as pd
import requests
from tqdm import tqdm

from settings import raw_csv_names


def missing_files(path: str, filenames: List[str]) -> bool:
    """
    Check if all file specified inside filenames list exists.
    :param path: the folder to work in
    :param filenames: list of filenames with extension
    :return: False if all filenames exists, True if only one missing
    """
    for filename in filenames:
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            return True
    return False


def gunzip(src_filepath: str, dest_filepath: str, block_size: int = 65536) -> None:
    with gzip.open(src_filepath, 'rb') as s_file, open(dest_filepath, 'wb') as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)


def tsv_to_csv(filepath: str):
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8', dtype='object')
    path_no_ext = os.path.splitext(filepath)[0].replace('.', '-')
    os.remove(filepath)
    df.to_csv(f'{path_no_ext}.csv', sep=',', index=False, encoding='utf-8')


def retrieve_csv(url: str, out_dir: str) -> bool:
    """
    Retrieve the zip/gz file containing the csv files if necessary and save them in local.
    :param url: the web URI to download the datasets
    :param out_dir: the folder in which to save the csv files
    :return: False if something went wrong during the download, True instead
    """
    to_download = missing_files(out_dir, raw_csv_names)

    if to_download:
        # At least one csv file missing, download zip
        filename = os.path.basename(url)
        filepath = os.path.join(out_dir, filename)
        if not os.path.exists(filepath):
            print(f'Some datasets missing, start download from {os.path.dirname(url)}')
            with requests.get(url, stream=True) as req:
                total_size = int(req.headers.get('Content-Length'))
                chunk_size = 1024
                wrote = 0
                with open(filepath, 'wb') as output:
                    # Prepare progress bar
                    for data in tqdm(
                            req.iter_content(chunk_size),
                            total=int(total_size // chunk_size),
                            unit='KB',
                            desc=filename,
                            file=sys.stdout
                    ):
                        wrote = wrote + len(data)
                        output.write(data)

            if total_size != 0 and wrote != total_size:
                print('Error, something went wrong during the download.')
                return False

            extracted = []
            ext_archive = os.path.splitext(filepath)[-1].lower()
            if ext_archive == '.zip':
                # TODO: make function for zip decompression
                # Open file zip and extract csv files inside out_dir
                with ZipFile(filepath, 'r') as archive:
                    for info in archive.infolist():
                        info.filename = os.path.basename(info.filename)
                        if info.filename != '':
                            ext = os.path.splitext(info.filename)[-1].lower()
                            if ext == '.csv':
                                extracted.append(info.filename)
                                archive.extract(info, out_dir)
            elif ext_archive == '.gz':
                dest_filepath = os.path.splitext(filepath)[0]
                gunzip(filepath, dest_filepath)
                tsv_to_csv(dest_filepath)

            os.remove(filepath)
            print(f'Following datasets saved:')
            for filename in extracted:
                print(f'\t- {filename}')
    else:
        print('No needed to download datasets.')
    print('-' * 30)

    return True


def request_features_tmdb(url: str, movie_id: int, tmdb_id: float, features: set) -> pd.DataFrame:
    """
    Query the TMDB Api to retrieve more features (budget, revenue, runtime, etc.).
    :param url: the web URI to query the TMDB Api
    :param movie_id: the index of a film from MovieLens
    :param tmdb_id: the index of a film from TMDB
    :param features: the set of features to extract from the JSON response
    :return: DataFrame with a single sample
    """
    response = requests.get(url)
    status_code = response.status_code

    budget, revenue, runtime = 0, 0, 0

    # 200 = OK
    if status_code == 200:
        json_response = response.json()
        keys = json_response.keys()
        if features.issubset(set(keys)):
            budget = json_response['budget']
            revenue = json_response['revenue']
            runtime = json_response['runtime']

    return pd.DataFrame({
        'movieId': [movie_id],
        'tmdbId': [tmdb_id],
        'budget': [budget],
        'revenue': [revenue],
        'runtime': [runtime]
    })
