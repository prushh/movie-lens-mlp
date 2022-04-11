import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.utils.util import missing_files
from src.utils.wrapper import convert_to, fill_na, drop, drop_na, extract_stat_feature, reset_index, rename, apply, \
    replace
from src.utils.const import RAW_DIR, EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, DROP, INTERIM_CSV_NAMES


def movies_processing(filepath: str) -> pd.DataFrame:
    def extract_year_from_title(df: pd.DataFrame) -> pd.DataFrame:
        regex = '.*\((\d{4})\).*'
        df['year'] = df['title'].str.extract(pat=regex, expand=False)
        return df

    def extract_title_length(df: pd.DataFrame) -> pd.DataFrame:
        df['title_length'] = df['title'].str.len()
        return df

    def encode_genre(df: pd.DataFrame) -> pd.DataFrame:
        genres = df['genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        encoded_genre = pd.DataFrame(
            mlb.fit_transform(genres),
            index=df['movieId'],
            columns=mlb.classes_
        )
        df = pd.merge(df, encoded_genre, on='movieId', how='inner')
        return df

    def remove_no_genres(df: pd.DataFrame) -> pd.DataFrame:
        df_no_genre = df[df['(no genres listed)'] == 1].index
        df.drop(index=df_no_genre, inplace=True)
        if DROP:
            print(f'Number of films without any genres to be dropped: {df_no_genre.shape[0]}')
        return df

    if os.path.exists(filepath):
        movies = pd.read_csv(
            filepath,
            encoding='utf-8',
            dtype={'movieId': 'uint16', 'year': 'float32', 'title_length': 'int32'}
        )
        return movies

    movies = pd.read_csv(
        os.path.join(RAW_DIR, 'movies.csv'),
        encoding='utf-8',
        dtype={'movieId': 'uint16', 'title': 'string', 'genres': 'category'}
    )

    movies = movies. \
        pipe(extract_year_from_title). \
        pipe(convert_to, 'year', 'float32'). \
        pipe(fill_na, 'year', 'median'). \
        pipe(extract_title_length). \
        pipe(convert_to, 'title_length', 'int32'). \
        pipe(encode_genre). \
        pipe(remove_no_genres). \
        pipe(drop, ['title', 'genres', '(no genres listed)'])

    movies.to_csv(filepath, encoding='utf-8', index=False)
    return movies


def tags_processing(filepath: str) -> pd.DataFrame:
    if os.path.exists(filepath):
        tags = pd.read_csv(
            filepath,
            encoding='utf-8',
            dtype={'movieId': 'uint16', 'tag_count': 'int32'}
        )
        return tags

    tags = pd.read_csv(
        os.path.join(RAW_DIR, 'tags.csv'),
        encoding='utf-8',
        usecols=['movieId', 'tag'],
        dtype={'movieId': 'uint16', 'tag': 'string'}
    )

    tags = tags. \
        pipe(drop_na). \
        pipe(extract_stat_feature, ['movieId'], 'tag', ['count']). \
        pipe(reset_index). \
        pipe(rename, {'count': 'tag_count'}). \
        pipe(convert_to, 'tag_count', 'int32'). \
        pipe(fill_na, 'tag_count', 'zero')

    tags.to_csv(filepath, encoding='utf-8', index=False)
    return tags


def ratings_processing(filepath: str) -> pd.DataFrame:
    if os.path.exists(filepath):
        ratings = pd.read_csv(
            filepath,
            encoding='utf-8',
            dtype={'movieId': 'uint16', 'rating_count': 'int32', 'rating_mean': 'float32'}
        )
        return ratings

    ratings = pd.read_csv(
        os.path.join(RAW_DIR, 'ratings.csv'),
        encoding='utf-8',
        usecols=['movieId', 'rating'],
        dtype={'movieId': 'uint16', 'rating': 'float32'}
    )

    ratings = ratings. \
        pipe(extract_stat_feature, ['movieId'], 'rating', ['count', 'mean']). \
        pipe(reset_index). \
        pipe(rename, {'count': 'rating_count', 'mean': 'rating_mean'}). \
        pipe(convert_to, 'rating_count', 'int32')

    ratings.to_csv(filepath, encoding='utf-8', index=False)
    return ratings


def imdb_processing(filepath: str) -> pd.DataFrame:
    if os.path.exists(filepath):
        imdb = pd.read_csv(
            filepath,
            encoding='utf-8',
            usecols=['tconst', 'runtime'],
            dtype={'tconst': 'string', 'runtime': 'float32'}
        )
        return imdb

    imdb = pd.read_csv(
        os.path.join(EXTERNAL_DIR, 'title-basics.csv'),
        encoding='utf-8',
        usecols=['tconst', 'runtimeMinutes'],
        dtype={'tconst': 'string'}
    )

    imdb = imdb. \
        pipe(replace, 'runtimeMinutes', '([\\]*[a-zA-Z|\-]+)', np.nan). \
        pipe(convert_to, 'runtimeMinutes', 'float32'). \
        pipe(rename, {'runtimeMinutes': 'runtime'})

    imdb.to_csv(filepath, encoding='utf-8', index=False)
    return imdb


def tmdb_processing(filepath: str) -> pd.DataFrame:
    def extract_correct_runtime(df: pd.DataFrame) -> pd.DataFrame:
        df['runtime'] = df['runtime_x'].mask((df['runtime_x'].isna()) | (df['runtime_x'] == 0), df['runtime_y'])
        return df

    if os.path.exists(filepath):
        tmdb = pd.read_csv(
            filepath,
            encoding='utf-8',
            usecols=['movieId', 'tmdbId', 'imdb_id', 'runtime'],
            dtype={'movieId': 'uint16', 'imdb_id': 'string', 'tmdbId': 'float32', 'runtime': 'float32'}
        )
        return tmdb

    imdb = pd.read_csv(
        os.path.join(INTERIM_DIR, 'title-basics.csv'),
        encoding='utf-8',
        usecols=['tconst', 'runtime'],
        dtype={'tconst': 'string', 'runtime': 'float32'}
    )

    tmdb = pd.read_csv(
        os.path.join(EXTERNAL_DIR, 'tmdb.csv'),
        encoding='utf-8',
        usecols=['movieId', 'tmdbId', 'imdb_id', 'runtime'],
        dtype={'movieId': 'uint16', 'imdb_id': 'string', 'tmdbId': 'float32', 'runtime': 'float32'}
    )

    tmdb = tmdb. \
        pipe(pd.merge, imdb, how='left', left_on='imdb_id', right_on='tconst'). \
        pipe(extract_correct_runtime). \
        pipe(fill_na, 'runtime', 'median'). \
        pipe(drop, ['tconst', 'runtime_x', 'runtime_y'])

    tmdb.to_csv(filepath, encoding='utf-8', index=False)
    return tmdb


def preprocessing() -> pd.DataFrame:
    some_interim_missing = missing_files(INTERIM_DIR, INTERIM_CSV_NAMES)

    filepath = os.path.join(INTERIM_DIR, 'movies.csv')
    movies = movies_processing(filepath)

    filepath = os.path.join(INTERIM_DIR, 'tags.csv')
    tags = tags_processing(filepath)

    filepath = os.path.join(INTERIM_DIR, 'ratings.csv')
    ratings = ratings_processing(filepath)

    filepath = os.path.join(INTERIM_DIR, 'title-basics.csv')
    _ = imdb_processing(filepath)

    filepath = os.path.join(INTERIM_DIR, 'tmdb.csv')
    tmdb = tmdb_processing(filepath)

    filepath = os.path.join(PROCESSED_DIR, 'final.csv')
    if os.path.exists(filepath) and not some_interim_missing:
        final = pd.read_csv(
            filepath,
            encoding='utf-8'
        )
        return final

    final = movies. \
        pipe(pd.merge, ratings, on='movieId', how='inner'). \
        pipe(pd.merge, tags, on='movieId', how='left'). \
        pipe(pd.merge, tmdb, on='movieId', how='inner')

    final.to_csv(filepath, encoding='utf-8', index=False)
    return final
