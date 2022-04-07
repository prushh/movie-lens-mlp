import os
from typing import List

import pandas as pd

from src.utils.const import DATA_DIR


def drop(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df.drop(columns=columns, inplace=True)
    return df


def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    return df


def copy_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


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
    return movies_tags_count


def preprocessing():
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    movies = pd.read_csv(os.path.join(RAW_DIR, 'movies.csv'), encoding='utf-8')
    links = pd.read_csv(os.path.join(RAW_DIR, 'links.csv'), encoding='utf-8')
    tags = pd.read_csv(os.path.join(RAW_DIR, 'tags.csv'), encoding='utf-8')
    ratings = pd.read_csv(os.path.join(RAW_DIR, 'ratings.csv'), encoding='utf-8')

    tags_clean = tags. \
        pipe(copy_df). \
        pipe(drop, ['userId', 'timestamp']). \
        pipe(drop_na)
