import os

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from settings import MOVIE_LENS_URL, DATASETS_DIR, YEAR_ENCODING, TMDB_URL
from utility import retrieve_csv, fill_budget_revenue


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

    if not retrieve_csv(MOVIE_LENS_URL, DATASETS_DIR):
        return 1

    # overview(DATASETS_DIR)

    # Links table never used
    movies = pd.read_csv('datasets/movies.csv', encoding='utf-8')
    links = pd.read_csv('datasets/links.csv', encoding='utf-8')
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

    # One-Hot Encoding for genres column inside movies dataframe
    genres = movies['genres'].str.split('|')
    mlb = MultiLabelBinarizer()
    encoded_genre = pd.DataFrame(
        mlb.fit_transform(genres),
        index=movies['movieId'],
        columns=mlb.classes_
    )
    movies = pd.merge(movies, encoded_genre, on='movieId', how='inner')
    # Drop title and genres columns
    movies.drop(columns=['title', 'genres'], inplace=True)

    # Drop timestamp and userId columns
    tags.drop(columns=['userId', 'timestamp'], inplace=True)
    # Drop all the rows that doesn't have the tag
    tags.dropna(inplace=True)

    movies_tags = pd.merge(movies, tags, on='movieId', how='left')
    movies_tags_count = pd.DataFrame(
        movies_tags.groupby('movieId', as_index=False)['tag']
            .count()
            .rename(columns={'tag': 'tag_count'})
    )
    movies = pd.merge(movies, movies_tags_count, on='movieId', how='inner')

    # Drop timestamp and userId column
    ratings.drop(columns=['userId', 'timestamp'], inplace=True)

    # calculate mean rate and number of rate for every movieId
    rating_mean = pd.DataFrame(
        ratings.groupby(['movieId'], as_index=False)['rating']
            .mean()
            .rename(columns={'rating': 'rating_mean'})
    )
    rating_count = pd.DataFrame(
        ratings.groupby(['movieId'], as_index=False)['rating']
            .count()
            .rename(columns={'rating': 'rating_count'})
    )
    rating_mean_count = pd.merge(rating_mean, rating_count, on='movieId', how='inner')

    # Merge the movies dataframe with the new rating_mean and rating_count columns
    movies = pd.merge(movies, rating_mean_count, on='movieId', how='inner')

    if YEAR_ENCODING:
        # One-Hot Encoding for year column inside movies dataframe
        movies_year = movies.set_index('movieId')['year']
        one_hot_encoded_year = pd.get_dummies(movies_year)

        # Add column of missing years with 0 since the year could be seen as categorical
        years_range = range(movies['year'].min(), movies['year'].max() + 1)
        for idx in years_range:
            if idx not in one_hot_encoded_year.columns:
                one_hot_encoded_year[idx] = 0

        movies = pd.merge(movies, one_hot_encoded_year, on='movieId', how='inner')
        movies.drop(columns='year', inplace=True)

    movies = pd.merge(movies, links, on='movieId', how='left')
    movies.dropna(inplace=True)

    movies_links = pd.merge(movies['movieId'], links, on='movieId', how='left')
    movies_links.dropna(inplace=True)

    net_value = pd.DataFrame()
    for (movie_id, _, tmdb_id) in movies_links.itertuples(name='Links', index=False):
        url = os.path.join(TMDB_URL, str(tmdb_id))
        net_value = pd.concat(
            [
                net_value,
                fill_budget_revenue(url, movie_id, tmdb_id)
            ],
            ignore_index=True)

        net_value.to_csv('datasets/net-value.csv', encoding='utf-8')

    return 0


if __name__ == '__main__':
    exit(main())
