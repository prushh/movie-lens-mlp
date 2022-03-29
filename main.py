import os

import pandas as pd

from settings import DATASETS_URL, DATASETS_DIR
from utility import retrieve_csv


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

    if not retrieve_csv(DATASETS_URL, DATASETS_DIR):
        return 1

    # overview(DATASETS_DIR)

    # Links table never used
    movies = pd.read_csv('datasets/movies.csv', encoding='utf-8')
    tags = pd.read_csv('datasets/tags.csv', encoding='utf-8')
    ratings = pd.read_csv('datasets/ratings.csv', encoding='utf-8')
    genome_tags = pd.read_csv('datasets/genome-tags.csv', encoding='utf-8')
    genome_scores = pd.read_csv('datasets/genome-scores.csv', encoding='utf-8')

    # Create column year from title
    movies['year'] = movies['title'].str.extract('.*\((\d+)\).*', expand=False)
    # Drop all the rows that doesn't have the year in the title
    movies.dropna(inplace=True)

    # Create column title_length from title
    movies['title_length'] = movies['title'].str.len()
    movies['title_length'] = movies['title_length'].astype('float32')
    # Drop title column
    movies.drop(columns='title', inplace=True)

    # One-Hot Encoding for genres column inside movies dataframe
    split_genre = movies.set_index('movieId')['genres'].str.split('|', expand=True).stack()
    one_hot_encoded_genre = pd.get_dummies(split_genre, dtype='float32').groupby(level=0).sum()
    movies = pd.merge(movies, one_hot_encoded_genre, on='movieId', how='inner')
    # Drop genres column from movies dataframe
    movies.drop(columns='genres', inplace=True)

    # Drop timestamp column
    tags.drop(columns='timestamp', inplace=True)
    # Drop userId column from tags dataframe
    tags.drop(columns='userId', inplace=True)
    # Drop all the rows that doesn't have the tag
    tags.dropna(inplace=True)

    movies_tags = pd.merge(movies, tags, on='movieId', how='left')
    movies_tags_count = pd.DataFrame(
        movies_tags.groupby('movieId', as_index=False)['tag']
            .count()
            .rename(columns={'tag': 'tag_count'})
    )
    movies = pd.merge(movies, movies_tags_count, on='movieId', how='inner')

    # Drop timestamp column
    ratings.drop(columns='timestamp', inplace=True)

    return 0


if __name__ == '__main__':
    exit(main())
