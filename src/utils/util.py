import gzip
import os
import sys
import zipfile
from typing import List, Set

import pandas as pd
import requests
from tqdm import tqdm


def overview_data(df: pd.DataFrame) -> None:
    print(f'Shape: {df.shape}')
    print(f'Columns: {df.columns.values}')


def output_datasets(extracted: List[str]) -> None:
    """
    Print all elements of extracted list
    :param extracted: the list that contains extracted files
    :return: None
    """
    print(f'Following data saved:')
    for filename in extracted:
        print(f'\t- {filename}')


def missing_files(path: str, filenames: List[str]) -> bool:
    """
    Check if all file specified inside filenames list exists
    :param path: the folder to work in
    :param filenames: list of filenames with extension
    :return: False if all filenames exists, True if only one missing
    """
    for filename in filenames:
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            return True
    return False


def get_missing_files(path: str, filenames: List[str]) -> List[str]:
    """
    Check if all file specified inside filenames list exists and
    return the missing files
    :param path: the folder to work in
    :param filenames: list of filenames with extension
    :return: a list with missing files
    """
    missing = []
    for filename in filenames:
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            missing.append(filename)
    return missing


def tsv_to_csv(filepath: str) -> str:
    """
    Given a filepath of .tsv file convert it to .csv and remove the old one
    :param filepath: the filepath to the .tsv file
    :return: the filename of the .csv file
    """
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8', dtype='object')
    path_no_ext = os.path.splitext(filepath)[0].replace('.', '-')
    os.remove(filepath)

    path_ext = f'{path_no_ext}.csv'
    df.to_csv(path_ext, sep=',', index=False, encoding='utf-8')
    filename = os.path.basename(path_ext)
    return filename


def csv_to_tsv_gz_ext(file_csv_names: List[str]) -> List[str]:
    """
    Remove .csv extension, add .tsv.gz and replace hyphen with dot
    :param file_csv_names: the list with .csv files
    :return: the new list with file .tsv.gz and no hyphen
    """
    file_tsv_names = []
    for filename in file_csv_names:
        no_ext = os.path.splitext(filename)[0]
        no_hyphen = no_ext.replace('-', '.')
        filename = f'{no_hyphen}.tsv.gz'
        file_tsv_names.append(filename)
    return file_tsv_names


def download_file(url: str, filepath: str, filename: str) -> bool:
    """
    Download a file specified by the url, it also shows a progress bar
    :param url: the web URI to the resource
    :param filepath: the path of the file to be downloaded
    :param filename: the name of the file to be downloaded
    :return: False if something went wrong during the download or True instead
    """
    with requests.get(url, stream=True) as req:
        total_size = int(req.headers.get('Content-Length'))
        chunk_size = 1024
        unit = 'KB'
        wrote = 0
        with open(filepath, 'wb') as output:
            # Prepare progress bar
            for data in tqdm(
                    req.iter_content(chunk_size),
                    total=int(total_size // chunk_size),
                    unit=unit,
                    desc=filename,
                    file=sys.stdout
            ):
                wrote = wrote + len(data)
                output.write(data)

    if total_size != 0 and wrote != total_size:
        print('Error, something went wrong during the download.')
        return False
    return True


def gunzip(src_filepath: str, dest_filepath: str, block_size: int = 65536) -> None:
    """
    Decompress .gz file.
    :param src_filepath: the input .gz filepath
    :param dest_filepath: the output filepath
    :param block_size: the chuck size for the decompression
    :return: None
    """
    with gzip.open(src_filepath, 'rb') as s_file, open(dest_filepath, 'wb') as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)


def unzip(src_filepath: str, dest_filepath: str, ext_to_save: str = '.csv') -> List[str]:
    """
    Decompress .zip file and save only those with the extension specified by ext_to_save
    :param src_filepath: the input .zip filepath
    :param dest_filepath: the output filepath where extract the archive
    :param ext_to_save: the extension of the files to save, default value is .csv
    :return: list of all extracted files
    """
    extracted = []
    with zipfile.ZipFile(src_filepath, 'r') as archive:
        for info in archive.infolist():
            info.filename = os.path.basename(info.filename)
            if info.filename != '':
                ext = os.path.splitext(info.filename)[-1].lower()
                if ext == ext_to_save:
                    extracted.append(info.filename)
                    archive.extract(info, dest_filepath)
    return extracted


def request_features_tmdb(url: str, movie_id: int, tmdb_id: float, features: Set[str]) -> pd.DataFrame:
    """
    Query the TMDB API to retrieve specified features
    :param url: the web URI to query the TMDB Api
    :param movie_id: the index of a film from MovieLens
    :param tmdb_id: the index of a film from TMDB
    :param features: the list with the features to extract
    :return: DataFrame with a single sample
    """
    response = requests.get(url)
    status_code = response.status_code

    data = {
        'movieId': [movie_id],
        'tmdbId': [tmdb_id]
    }
    for key in features:
        data[key] = ['NaN']

    # 200 = OK
    if status_code == 200:
        json_response = response.json()
        keys = json_response.keys()
        if features.issubset(set(keys)):
            for key in features:
                data[key] = [json_response[key]]
        else:
            for key in features:
                data[key] = ['NaN']

    return pd.DataFrame(data)
