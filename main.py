import argparse
import warnings

from src.data.preprocessing import preprocessing
from src.data.acquisition import retrieve_datasets
from src.models.network.mlp import mlp
from src.models.sklearn_models import fit_model
from src.utils.util_models import fix_random

warnings.filterwarnings('ignore', category=UserWarning)


def main() -> int:
    fix_random(42)
    if not retrieve_datasets():
        return 1

    final = preprocessing()

    if args.model == 'mlp':
        mlp(final, args.easy, args.best)
    else:
        fit_model(final, args.model, args.easy, args.best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data Analytics project using MovieLens dataset.',
        usage='%(prog)s model [--easy | --best]'
    )

    parser.add_argument(
        'model', default='none', type=str,
        choices=['mlp', 'tree_based', 'svm', 'naive_bayes'],
        help='the name of the model'
    )
    parser.add_argument(
        '-e', '--easy', default=False,
        action='store_true',
        help='demo purpose, use only one random configuration for hyperparams'
    )
    parser.add_argument(
        '-b', '--best', default=False,
        action='store_true',
        help='use the best training configuration'
    )

    args = parser.parse_args()
    if args.easy and args.best:
        parser.error('specify only --easy or --best, not both together')

    exit(main())
