import os.path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.utils.wrapper import convert_to, fill_na, drop
from src.utils.const import RAW_DIR, EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, DROP


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


def preprocessing() -> pd.DataFrame:
    filepath = os.path.join(PROCESSED_DIR, 'final.csv')
    if os.path.exists(filepath):
        final = pd.read_csv(os.path.join(filepath, 'final.csv'), encoding='utf-8')
        return final

    movies = pd.read_csv(
        RAW_DIR,
        'movies.csv',
        encoding='utf-8',
        dtype={'title': 'string', 'genres': 'category'}
    )

    movies = movies. \
        pipe(extract_year_from_title). \
        pipe(convert_to, 'year', 'float32'). \
        pipe(fill_na, 'year', True). \
        pipe(extract_title_length). \
        pipe(encode_genre). \
        pipe(remove_no_genres). \
        pipe(drop, ['title', 'genres'])

    return pd.DataFrame()
