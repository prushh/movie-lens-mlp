import os
from string import Template

MOVIE_LENS_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
TMDB_API_URL = Template('https://api.themoviedb.org/3/movie/${tmdb_id}?api_key=${api_key}')
IMDB_URL = 'https://datasets.imdbws.com/'

DATASETS_DIR = 'datasets'

folders_name = [
    'external',
    'interim',
    'processed',
    'raw'
]

raw_csv_names = [
    'genome-scores.csv',
    'genome-tags.csv',
    'links.csv',
    'movies.csv',
    'ratings.csv',
    'tags.csv'
]

external_tsv_names = [
    'title.basics.tsv.gz'
]

external_imdb_csv_names = [
    f"{os.path.splitext(os.path.splitext(filename)[0])[0].replace('.', '-')}.csv" for filename in external_tsv_names
]

external_tmdb_csv_names = [
    'tmdb-features.csv'
]

PLOT = True
