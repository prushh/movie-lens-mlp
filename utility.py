import gzip
import os
import sys
import zipfile
from typing import List, Set

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from settings import TMDB_API_URL


def output_datasets(extracted: List[str]) -> None:
    """
    Print all elements of extracted list.
    :param extracted: the list that contains extracted files
    :return: None
    """
    print(f'Following datasets saved:')
    for filename in extracted:
        print(f'\t- {filename}')


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


def get_missing_files(path: str, filenames: List[str]) -> List[str]:
    """
    Check if all file specified inside filenames list exists and
    return the missing files.
    :param path: the folder to work in
    :param filenames: list of filenames with extension
    :return: a list with missing files
    """
    missing = []
    for filename in filenames:
        base_tsv = os.path.splitext(filename)[0]
        base = os.path.splitext(base_tsv)[0].replace('.', '-')
        filepath = os.path.join(path, f'{base}.csv')
        if not os.path.exists(filepath):
            missing.append(filename)
    return missing


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


def unzip(src_filepath: str, dest_filepath: str) -> list:
    """
    Decompress .zip file.
    :param src_filepath: the input .zip filepath
    :param dest_filepath: the output filepath where extract the archive
    :return: list of all extracted files
    """
    extracted = []
    with zipfile.ZipFile(src_filepath, 'r') as archive:
        for info in archive.infolist():
            info.filename = os.path.basename(info.filename)
            if info.filename != '':
                ext = os.path.splitext(info.filename)[-1].lower()
                if ext == '.csv':
                    extracted.append(info.filename)
                    archive.extract(info, dest_filepath)
    return extracted


def tsv_to_csv(filepath: str) -> str:
    """
    Given a filepath of .tsv file convert it to .csv and remove the old one.
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


def download_file(url: str, filepath: str, filename: str) -> bool:
    """
    Download a file specified by the url, it also shows a progress bar.
    :param url: the web URI to the resource
    :param filepath: the path of the file to be downloaded
    :param filename: the name of the file to be downloaded
    :return: False if something went wrong during the download or True instead
    """
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
    return True


def retrieve_movie_lens(url: str, out_dir: str, file_csv_names: List) -> bool:
    """
    Retrieve, if necessary, MovieLens datasets specified in the file_csv_names list.
    :param url: the web URI to the resource
    :param out_dir: the directory where the files are saved
    :param file_csv_names: the list with all the files to check
    :return: False if something went wrong during the download or True instead
    """
    to_download = missing_files(out_dir, file_csv_names)

    if to_download:
        # At least one csv file missing, download zip
        filename = os.path.basename(url)
        filepath = os.path.join(out_dir, filename)
        if not os.path.exists(filepath):
            print(f'Some raw MovieLens datasets missing, start download from {os.path.dirname(url)}')
            if not download_file(url, filepath, filename):
                return False

        extracted = unzip(filepath, out_dir)
        os.remove(filepath)
        output_datasets(extracted)
    else:
        print('No needed to download raw MovieLens datasets.')
        print('-' * 30)

    return True


def retrieve_imdb(url: str, out_dir: str, file_tsv_names: List[str]) -> bool:
    """
    Retrieve, if necessary, IMDB datasets specified in the file_tsv_names list.
    :param url: the web URI to the resource
    :param out_dir: the directory where the files are saved
    :param file_tsv_names: the list with all the files to check
    :return: False if something went wrong during the download or True instead
    """
    missing = get_missing_files(out_dir, file_tsv_names)

    if missing:
        extracted = []
        print(f'Some external IMDB datasets missing, start download from {os.path.dirname(url)}')
        for filepath in missing:
            filename = os.path.basename(filepath)
            if not os.path.exists(filepath):
                file_url = os.path.join(url, filename)
                filepath = os.path.join(out_dir, filename)
                if not download_file(file_url, filepath, filename):
                    return False

            dest_filepath = os.path.splitext(filepath)[0]
            gunzip(filepath, dest_filepath)
            converted = tsv_to_csv(dest_filepath)
            extracted.append(converted)

            os.remove(filepath)

        output_datasets(extracted)
    else:
        print('No needed to download external IMDB datasets.')
        print('-' * 30)

    return True


def request_features_tmdb(url: str, movie_id: int, tmdb_id: float, features: Set[str]) -> pd.DataFrame:
    """
    Query the TMDB API to retrieve specified features.
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


def retrieve_tmdb(df: pd.DataFrame, out_dir: str, file_csv_names: List, features: Set[str],
                  filename: str = 'tmdb-features.csv') -> bool:
    """
    If not exists create a .csv that contains new features, retrieved by API requests to TMDB service.
    :param df: the DataFrame that contains movies_id and tmdb_id
    :param out_dir: the folder in which to save the csv files
    :param file_csv_names: the list of all files to check the existence
    :param features: the list with the features to extract
    :param filename: the name of the new .csv file
    :return: False if something went wrong during the requests, True instead
    """
    to_download = missing_files(out_dir, file_csv_names)

    if to_download:
        load_dotenv()
        token = os.environ.get('TMDB_API_KEY')

        url = TMDB_API_URL.substitute(tmdb_id='', api_key='')
        print(f"Some external TMDB datasets missing, start API requests from{os.path.dirname(url)}")

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
                print('Error, something went wrong during the API requests.')
                return False
            else:
                tmdb_features = pd.concat([tmdb_features, sample], ignore_index=True)

        tmdb_features_path = os.path.join(out_dir, filename)
        tmdb_features.to_csv(tmdb_features_path, index=False, encoding='utf-8')
        output_datasets([filename])
    else:
        print('No needed to download external TMDB datasets.')
        print('-' * 30)

    return True
