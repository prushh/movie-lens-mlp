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

    if args.model == 'mlp':
        mlp(final, args.random, args.best)
    else:
        fit_model(final, args.model, args.random, args.best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data Analytics project using MovieLens dataset.',
        usage='%(prog)s model [--random | --best]'
    )

    parser.add_argument(
        'model', default='none', type=str,
        choices=['mlp', 'tree_based', 'svm', 'naive_bayes'],
        help='the name of the model'
    )
    parser.add_argument(
        '-r', '--random', default=False,
        action='store_true',
        help='demo purpose, use only one random configuration for hyperparams'
    )
    parser.add_argument(
        '-b', '--best', default=False,
        action='store_true',
        help='use the best training configuration'
    )

    args = parser.parse_args()
    if args.random and args.best:
        parser.error('specify only --easy or --best, not both together')

    exit(main())
