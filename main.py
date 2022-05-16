from src.data.preprocessing import preprocessing
from src.data.acquisition import retrieve_datasets
from src.models.sklearn_models import fit_model


def main() -> int:
    if not retrieve_datasets():
        return 1

    final = preprocessing()

    models = ['tree_based', 'svm', 'naive_bayes']

    for model_group in models:
        fit_model(final, model_group)


if __name__ == '__main__':
    exit(main())
