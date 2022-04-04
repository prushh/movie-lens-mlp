import os.path
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

from settings import raw_csv_names, external_imdb_csv_names, external_tmdb_csv_names, PLOT, DATASETS_DIR

# sns.set_style('darkgrid', {'font.size': 18})
plt.rcParams.update({'figure.figsize': (16, 10), 'figure.dpi': 100, 'font.size': 18})


def import_df(file_csv_names: List[str], path: str) -> {}:
    dict_df = {}
    for name in file_csv_names:
        name_no_ext = os.path.splitext(name)[0]
        path_dataset = os.path.join(path, name)
        if name_no_ext != 'links':
            dict_df[name_no_ext] = pd.read_csv(path_dataset, encoding='utf-8')
        else:
            dict_df[name_no_ext] = pd.read_csv(path_dataset, encoding='utf-8', dtype={'imdbId': 'string'})

    return dict_df


def rating_preprocessing(ratings: pd.DataFrame) -> pd.DataFrame:
    # Drop timestamp and userId column
    local_ratings = ratings.drop(columns=['userId', 'timestamp'])
    # calculate mean rate and number of rate for every movieId
    rating_mean = pd.DataFrame(
        local_ratings.groupby(['movieId'], as_index=False)['rating']
            .mean()
            .rename(columns={'rating': 'rating_mean'})
    )
    rating_count = pd.DataFrame(
        local_ratings.groupby(['movieId'], as_index=False)['rating']
            .count()
            .rename(columns={'rating': 'rating_count'})
    )
    rating_mean_count = pd.merge(rating_mean, rating_count, on='movieId', how='inner')

    return rating_mean_count


def tag_preprocessing(tags: pd.DataFrame) -> pd.DataFrame:
    # Drop timestamp and userId columns
    local_tags = tags.drop(columns=['userId', 'timestamp'])
    # Drop all the rows that doesn't have the tag
    local_tags.dropna(inplace=True)
    movies_tags_count = pd.DataFrame(
        local_tags.groupby('movieId', as_index=False)['tag']
            .count()
            .rename(columns={'tag': 'tag_count'})
    )
    # TODO add merge with movies and fillna with 0
    return movies_tags_count


def movies_preprocessing(movies: pd.DataFrame) -> pd.DataFrame:
    local_movies = extract_year(movies)
    local_movies['title_length'] = movies['title'].str.len()
    local_movies = sep_genres_movies(local_movies)
    # Drop title and genres columns after encoding
    local_movies.drop(columns=['title', 'genres'], inplace=True)
    return local_movies


def sep_genres_movies(movies: pd.DataFrame) -> pd.DataFrame:
    # One-Hot Encoding for genres column inside movies dataframe
    genres = movies['genres'].str.split('|')
    mlb = MultiLabelBinarizer()
    encoded_genre = pd.DataFrame(
        mlb.fit_transform(genres),
        index=movies['movieId'],
        columns=mlb.classes_
    )
    movies = pd.merge(movies, encoded_genre, on='movieId', how='inner')
    return movies


def extract_runtime(imdb: pd.DataFrame, tmdb: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    missing_runtime = tmdb[(tmdb['runtime'] == 0) | (tmdb['runtime'].isna())].copy()

    # Drop from the tmdb all the rows that are nan since are saved in missing_runtime
    tmdb.set_index('movieId', inplace=True)
    tmdb.drop(index=missing_runtime['movieId'], inplace=True)

    # Gathering runtime from IMDB thanks to tconst
    link_missing_runtime = pd.merge(missing_runtime['movieId'], links, on='movieId', how='left')
    link_missing_runtime.drop(columns='tmdbId', inplace=True)
    link_missing_runtime.rename(columns={'imdbId': 'tconst'}, inplace=True)
    link_missing_runtime['tconst'] = link_missing_runtime['tconst'].apply(lambda x: f"tt{x}")
    imdb.rename(columns={'runtimeMinutes': 'runtime'}, inplace=True)
    imdb_movie_id = pd.merge(link_missing_runtime, imdb, on='tconst', how='left')

    # Removing '\\N' and other wrong values from the runtime column
    imdb_movie_id['runtime'].replace(regex='([\\]*[a-zA-Z|\-]+)', value=np.nan, inplace=True)
    imdb_movie_id['runtime'] = imdb_movie_id['runtime'].astype('float32')

    # Merging the gathered runtime with the missing_runtime dataframe
    imdb_movie_id = imdb_movie_id[['movieId', 'runtime']]
    missing_runtime.drop(columns='runtime', inplace=True)
    missing_runtime = pd.merge(missing_runtime, imdb_movie_id, on='movieId', how='inner')
    # Merge the missing_runtime with the gathered values to the tmdb dataset
    tmdb.reset_index(inplace=True)
    tmdb = tmdb.rename(columns={'index': 'movieId'})
    tmdb = pd.concat([tmdb, missing_runtime])
    runtime = tmdb['runtime'].dropna().astype('int32')
    if PLOT:
        plot_distribution(
            runtime,
            title='Frequency Histogram',
            x_label='Duration in minutes',
            y_label='Number of films',
            bins=50
        )
    median_runtime = runtime.median()
    tmdb['runtime'].fillna(median_runtime, inplace=True)

    return tmdb


def external_preprocessing(imdb: pd.DataFrame, tmdb: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    local_tmdb = extract_runtime(imdb, tmdb, links)
    return local_tmdb


def extract_year(movies: pd.DataFrame) -> pd.DataFrame:
    movies['year'] = movies['title'].str.extract('.*\((\d{4})\).*', expand=False)
    years = movies['year'].dropna().astype('int32')
    if PLOT:
        plot_distribution(
            years,
            title='Frequency Histogram',
            x_label='Release year',
            y_label='Number of films',
            bins=30
        )
    median_year = years.median()
    # Filling data missing
    movies['year'].fillna(median_year)

    # TODO: review minmax scaling if must be somewhere else
    # min-max scaling
    movie_year = movies['year'].astype('float32')
    year_min = movie_year.min()
    year_max = movie_year.max()
    movie_year -= year_min
    movie_year /= (year_max - year_min)
    movies['year'] = movie_year
    return movies


def plot_distribution(x_values: pd.Series, title: str, x_label: str, y_label: str, bins: int = 25):
    # Plot instance
    sns.histplot(
        data=x_values,
        bins=bins,
        kde=True,
        line_kws={"linewidth": 3})
    plt.gca().set(title=title, ylabel=y_label, xlabel=x_label)
    plt.show()


def merge_final_df(dict_all_df: Dict) -> pd.DataFrame:
    pass


def preprocessing() -> pd.DataFrame:
    all_csv_names = raw_csv_names + external_imdb_csv_names + external_tmdb_csv_names
    path = DATASETS_DIR
    dict_df = import_df(all_csv_names, path)
    dict_df['tmdb-features'] = external_preprocessing(dict_df['title-basics'], dict_df['tmdb-features'],
                                                      dict_df['links'])
    dict_df['tags'] = tag_preprocessing(dict_df['tags'])
    dict_df['ratings'] = rating_preprocessing(dict_df['ratings'])
    dict_df['movies'] = movies_preprocessing(dict_df['movies'])
    return pd.DataFrame()
