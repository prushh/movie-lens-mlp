from string import Template

MOVIE_LENS_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
TMDB_URL = 'https://www.themoviedb.org/movie/'
TMDB_API_URL = Template('https://api.themoviedb.org/3/movie/${tmdb_id}?api_key=${api_key}')

DATASETS_DIR = 'datasets'

csv_names = [
    'genome-scores.csv',
    'genome-tags.csv',
    'links.csv',
    'movies.csv',
    'ratings.csv',
    'tags.csv'
]

YEAR_ENCODING = False
LINKS_FLAG = True
