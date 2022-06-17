import argparse

from src.data.preprocessing import preprocessing
from src.data.acquisition import retrieve_datasets
from src.models.network.mlp import mlp
from src.models.sklearn_models import fit_model
from src.utils.util_models import fix_random


def main() -> int:
    fix_random(42)
    if not retrieve_datasets():
        return 1

    final = preprocessing()

    models = ['tree_based', 'svm', 'naive_bayes']

    # for model_group in models:
    #     fit_model(final, model_group)

    mlp(final)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='Data Analytics project using MovieLens dataset.',
    #     usage='%(prog)s model [--test MODE [-i FILENAME]]'
    # )
    #
    # parser.add_argument(
    #     'model', default='none', type=str,
    #     choices=['mlp', 'tree_based', 'svm', 'naive_bayes'],
    #     help='the name of the model'
    # )
    # parser.add_argument(
    #     '--test', default=False,
    #     action='store_true',
    #     help='test model specified by filename'
    # )
    # parser.add_argument(
    #     '-i', '--input', default='none', type=str,
    #     help='the output folder of the experiment'
    # )
    #
    # args = parser.parse_args()

    exit(main())
