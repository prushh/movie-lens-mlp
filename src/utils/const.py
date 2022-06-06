import os
from string import Template

MOVIE_LENS_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
TMDB_API_URL = Template('https://api.themoviedb.org/3/movie/${tmdb_id}?api_key=${api_key}')
TMDB_URL = 'https://github.com/prushh/movie-lens-mlp/releases/download/api-dataset/tmdb.csv'
IMDB_URL = 'https://datasets.imdbws.com/'

DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
FIGURE_DIR = os.path.join('reports', 'figures')
LOG_DIR = os.path.join('src', 'models', 'log')
NETWORK_RESULTS_DIR = os.path.join('src', 'models', 'network', 'results')

DATA_SUB_DIRS = [
    'external',
    'interim',
    'processed',
    'raw'
]

RAW_CSV_NAMES = [
    'genome-scores.csv',
    'genome-tags.csv',
    'links.csv',
    'movies.csv',
    'ratings.csv',
    'tags.csv'
]

EXTERNAL_IMDB_CSV_NAMES = [
    'title-basics.csv'
]

EXTERNAL_TMDB_CSV_NAMES = [
    'tmdb.csv'
]

INTERIM_PARQUET_NAMES = [
    'movies.parquet',
    'ratings.parquet',
    'tags.parquet',
    'title-basics.parquet',
    'tmdb.parquet'
]

NUM_BINS = 10
SEED = 42
USE_API = False
PLOT = False
