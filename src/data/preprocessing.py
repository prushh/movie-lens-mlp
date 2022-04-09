import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.utils.wrapper import convert_to, fill_na, drop, drop_na, extract_stat_feature, reset_index, rename, apply, \
    replace
from src.utils.const import RAW_DIR, EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, DROP


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

    movies = pd.read_csv(
        os.path.join(RAW_DIR, 'movies.csv'),
        encoding='utf-8',
        dtype={'movieId': 'uint16', 'title': 'string', 'genres': 'category'}
    )

    movies = movies. \
        pipe(extract_year_from_title). \
        pipe(convert_to, 'year', 'float32'). \
        pipe(fill_na, 'year', True). \
        pipe(extract_title_length). \
        pipe(convert_to, 'title_length', 'int32'). \
        pipe(encode_genre). \
        pipe(remove_no_genres). \
        pipe(drop, ['title', 'genres'])

    movies.to_csv(filepath, encoding='utf-8', index=False)
    return movies


def tags_processing(filepath: str) -> None:
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
        pipe(rename, {'count': 'tag_count'})

    tags.to_csv(filepath, encoding='utf-8', index=False)


def ratings_processing(filepath: str) -> None:
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


def links_processing(filepath: str) -> None:
    links = pd.read_csv(
        os.path.join(RAW_DIR, 'links.csv'),
        encoding='utf-8',
        usecols=['movieId', 'imdbId'],
        dtype={'movieId': 'uint16', 'imdbId': 'string'}
    )

    links = links. \
        pipe(apply, 'imdbId', lambda x: f'tt{x}'). \
        pipe(convert_to, 'imdbId', 'string')

    links.to_csv(filepath, encoding='utf-8', index=False)


def imdb_processing(filepath: str) -> None:
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


def tmdb_processing(filepath: str) -> None:
    def extract_correct_runtime(df: pd.DataFrame) -> pd.DataFrame:
        df['runtime'] = df['runtime_x'].mask((df['runtime_x'].isna()) | (df['runtime_x'] == 0), df['runtime_y'])
        return df

    imdb = pd.read_csv(
        os.path.join(INTERIM_DIR, 'title-basics.csv'),
        encoding='utf-8',
        usecols=['tconst', 'runtime'],
        dtype={'tconst': 'string'}
    )

    links = pd.read_csv(
        os.path.join(INTERIM_DIR, 'links.csv'),
        encoding='utf-8',
        dtype={'movieId': 'uint16', 'imdbId': 'string'}
    )

    tmdb = pd.read_csv(
        os.path.join(EXTERNAL_DIR, 'tmdb.csv'),
        encoding='utf-8',
        usecols=['movieId', 'tmdbId', 'runtime'],
        dtype={'movieId': 'uint16', 'tmdbId': 'float32', 'runtime': 'float32'}
    )

    tmdb = tmdb. \
        pipe(pd.merge, links, how='left', on='movieId'). \
        pipe(pd.merge, imdb, how='left', left_on='imdbId', right_on='tconst'). \
        pipe(extract_correct_runtime). \
        pipe(fill_na, 'runtime', True). \
        pipe(drop, ['tconst', 'runtime_x', 'runtime_y'])

    tmdb.to_csv(filepath, encoding='utf-8', index=False)


def preprocessing() -> pd.DataFrame:
    filepath = os.path.join(INTERIM_DIR, 'movies.csv')
    if not os.path.exists(filepath):
        # TODO: return df also in other functions
        movies = movies_processing(filepath)

    filepath = os.path.join(INTERIM_DIR, 'tags.csv')
    if not os.path.exists(filepath):
        tags_processing(filepath)

    filepath = os.path.join(INTERIM_DIR, 'ratings.csv')
    if not os.path.exists(filepath):
        ratings_processing(filepath)

    filepath = os.path.join(INTERIM_DIR, 'links.csv')
    if not os.path.exists(filepath):
        links_processing(filepath)

    filepath = os.path.join(INTERIM_DIR, 'title-basics.csv')
    if not os.path.exists(filepath):
        imdb_processing(filepath)

    filepath = os.path.join(INTERIM_DIR, 'tmdb.csv')
    if not os.path.exists(filepath):
        tmdb_processing(filepath)

    filepath = os.path.join(PROCESSED_DIR, 'final.csv')
    if not os.path.join(filepath):
        # TODO: read all interim .csv with correct type for columns
        # TODO: merge all (see preprocessing_old.py)
        pass

    return pd.DataFrame()
