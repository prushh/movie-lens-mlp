import os
import sys
from typing import List
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from settings import csv_names


def contains_number(value):
    return any([char.isdigit() for char in value])


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
            print(f'Following datasets saved:')
            for filename in extracted:
                print(f'\t- {filename}')
    else:
        print('No needed to download datasets.')
    print('-' * 30)

    return True


def fill_budget_revenue(url: str):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36'
    }
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    p_list = soup.findAll('section', attrs={'class': ['facts', 'left_column']})[0].select('p')
    budget = p_list[-2:-1][0].getText().split('$')[-1].replace(',', '')
    revenue = p_list[-1].getText().split('$')[-1].replace(',', '')
    if contains_number(budget):
        print(f'budget:{float(budget)}')
    else:
        print(0)
    if contains_number(revenue):
        print(f'revenue:{float(revenue)}')

