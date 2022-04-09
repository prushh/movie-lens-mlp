from src.data.preprocessing import preprocessing
from src.data.acquisition import retrieve_datasets


def main() -> int:
    if not retrieve_datasets():
        return 1

    preprocessing()


if __name__ == '__main__':
    exit(main())
