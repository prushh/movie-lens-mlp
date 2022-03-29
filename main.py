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
    if not retrieve_csv(DATASETS_URL, DATASETS_DIR):
        return 1

    # overview(DATASETS_DIR)

    # Links table never used
    # links = pd.read_csv('datasets/links.csv', encoding='utf-8')
    movies = pd.read_csv('datasets/movies.csv', encoding='utf-8')
    tags = pd.read_csv('datasets/tags.csv', encoding='utf-8')
    ratings = pd.read_csv('datasets/ratings.csv', encoding='utf-8')
    genome_tags = pd.read_csv('datasets/genome-tags.csv', encoding='utf-8')
    genome_scores = pd.read_csv('datasets/genome-scores.csv', encoding='utf-8')

    # Drop timestamp column from ratings and tags dataframes because they will not be used.
    ratings.drop(columns='timestamp', inplace=True)
    tags.drop(columns='timestamp', inplace=True)

    # One-Hot Encoding for genres column inside movies dataframe
    split_genre = movies.set_index('movieId')['genres'].str.split('|', expand=True).stack()
    one_hot_encoded = pd.get_dummies(split_genre).groupby(level=0).sum()
    movies = pd.merge(movies, one_hot_encoded, on='movieId', how='inner')
    # Drop genres column from movies dataframe
    movies.drop(columns='genres', inplace=True)

    # Save column year
    movies['year'] = movies['title'].str.extract('.*\((\d+)\).*', expand=False)
    # Drop all the rows that doesn't have the year in the title
    movies.dropna(inplace=True)
    movies['year'] = movies['year'].astype('int32')
    # Remove year from the title
    movies['title'] = movies['title'].str[:-7]
    # ratings.drop(columns='timestamp', inplace=True)
    mean_rating = pd.DataFrame(ratings.groupby(['movieId'])['rating'].mean())

    mean_rating.rename(columns={"rating": "mean_rating"}, inplace=True)
    number_rating = pd.DataFrame(ratings.groupby(['movieId']).count())
    number_rating.drop(columns='rating', inplace=True)
    number_rating.rename(columns={"userId": "number_of_ratings"}, inplace=True)
    number_mean_rating = pd.merge(mean_rating, number_rating, on='movieId', how='inner')
    # print(number_rating)

    movies = pd.merge(movies, number_mean_rating, on='movieId', how='inner')
    movies_year = movies.set_index('movieId')['year']
    one_hot_encoded = pd.get_dummies(movies_year)

    for idx in range(movies['year'].min(), movies['year'].max()):
        if idx not in one_hot_encoded.columns:
            one_hot_encoded[idx] = 0

    movies = pd.merge(movies, one_hot_encoded, on='movieId', how='inner')
    movies = movies.drop(columns='year')
    print(movies.columns)

    # pd.rename(columns={"A": "a", "B": "c"})
    return 0


if __name__ == '__main__':
    exit(main())
