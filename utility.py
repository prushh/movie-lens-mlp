import os
import sys
from zipfile import ZipFile

import requests
from tqdm import tqdm

from settings import csv_names


def missing_files(path: str, filenames: list[str]) -> bool:
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


def retrieve_csv(url: str, out_dir: str) -> bool:
    """
    Retrieve the zip file containing the csv files if necessary and save them in local.
    :param url: the web URI to download the datasets
    :param out_dir: the folder in which to save the csv files
    :return: False if something went wrong during the download, True instead
    """
    to_download = missing_files(out_dir, csv_names)

    if to_download:
        # At least one csv file missing, download zip
        filename = os.path.basename(url)
        filepath = os.path.join(out_dir, filename)
        if not os.path.exists(filepath):
            print('Datasets not found, start download...')
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
            # Open file zip and extract csv files inside out_dir
            with ZipFile(filepath, 'r') as archive:
                for info in archive.infolist():
                    info.filename = os.path.basename(info.filename)
                    if info.filename != '':
                        ext = os.path.splitext(info.filename)[-1].lower()
                        if ext == '.csv':
                            extracted.append(info.filename)
                            archive.extract(info, out_dir)

            os.remove(filepath)
            # TODO: pretty output message
            print(f'Following datasets saved: {extracted}')
    else:
        print('No needed to download datasets.')

    return True
