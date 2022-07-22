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
        # TODO: test inside mlp
        mlp(final, args.easy, args.test)
    else:
        fit_model(final, args.model, args.easy, args.input, args.test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data Analytics project using MovieLens dataset.',
        usage='%(prog)s model [--train -e] | [--test [-i FILENAME]]'
    )

    parser.add_argument(
        'model', default='none', type=str,
        choices=['mlp', 'tree_based', 'svm', 'naive_bayes'],
        help='the name of the model'
    )
    parser.add_argument(
        '--train', default=False,
        action='store_true',
        help='train specified model'
    )
    parser.add_argument(
        '-e', '--easy', default=False,
        action='store_true',
        help='demo purpose, reduce hyperparams space'
    )
    parser.add_argument(
        '--test', default=False,
        action='store_true',
        help='test model specified by filename'
    )
    parser.add_argument(
        '-i', '--input', default='none', type=str,
        help='the output folder of the experiment'
    )

    args = parser.parse_args()
    if args.train and args.test:
        parser.error('specify only --train or --test, not both together')

    if args.test and args.easy:
        parser.error('specify --easy only with --train flag')

    if args.train and not (args.input == 'none'):
        parser.error('specify --input only with --test flag')

    exit(main())
