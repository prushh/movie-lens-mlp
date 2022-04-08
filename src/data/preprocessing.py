import os

import pandas as pd
from src.utils.const import RAW_DIR, EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR


def preprocessing() -> pd.DataFrame:
    movies = pd.read_csv(os.path.join(RAW_DIR, 'movies.csv'), encoding='utf-8')
    links = pd.read_csv(os.path.join(RAW_DIR, 'links.csv'), encoding='utf-8')
    tags = pd.read_csv(os.path.join(RAW_DIR, 'tags.csv'), encoding='utf-8')
    ratings = pd.read_csv(os.path.join(RAW_DIR, 'ratings.csv'), encoding='utf-8')

    return pd.DataFrame()
