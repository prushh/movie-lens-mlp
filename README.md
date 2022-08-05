# Classification of MovieLens Tabular Data

Project for the course "Data Analytics" of the University of Bologna, A.Y. 2021/2022. In this project a data pipeline was implemented to predict the average mark of a film from its features, using Machine Learning techniques.

## Developers

- [Cotugno Giosuè](https://github.com/cotus997)
- [Davide Pruscini](https://github.com/prushh)

## Setup

To execute the script, [Python](https://www.python.org/) must be installed, and some external libraries must be downloaded and installed using the pip (or pip3) package manager:

```bash
pip install -r requirements.txt
```

We recommend the use of a virtual environment such as [conda](https://www.anaconda.com/products/distribution), for example, for package installation and project execution.

### Environment variable

The file .env.example must be renamed to .env and the single variable TMDB_API_KEY must be set to the respective key of [TMDB](https://www.themoviedb.org/documentation/api). You only need to specify it if you want to download the TMDB dataset via api calls.

## Usage

```console
python main.py -h

usage: main.py model [--easy | --best]

Data Analytics project using MovieLens dataset.

positional arguments:
  {mlp,tree_based,svm,naive_bayes}
                        the name of the model

options:
  -h, --help            show this help message and exit
  -e, --easy            demo purpose, use only one random configuration for hyperparams
  -b, --best            use the best training configuration
```

## Notebooks

The notebooks contain fundamental project parts that have been implemented for greater understanding. In order to avoid errors, we recommend running the notebooks in alphabetical order.


## Report

The report describing the various parts of the project from both an implementation and conceptual point of view is the following: [main.pdf](https://github.com/prushh/movie-lens-mlp/blob/main/reports/main.pdf)
