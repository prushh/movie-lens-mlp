from src.data.preprocessing import preprocessing
from src.data.acquisition import retrieve_datasets
from src.models.network.mlp import mlp


def main() -> int:
    if not retrieve_datasets():
        return 1

    final = preprocessing()

    mlp(final)


if __name__ == '__main__':
    exit(main())
