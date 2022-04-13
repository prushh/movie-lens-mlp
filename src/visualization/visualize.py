import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.utils.const import FIGURE_DIR

custom_params = {
    'figure.figsize': (16, 10),
    'lines.linewidth': 3,
    'axes.titlesize': 40,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
sns.set_theme(rc=custom_params)


def histogram(x_values: pd.Series, title: str, xlabel: str, ylabel: str, filename: str = '', **kwargs) -> None:
    sns.histplot(
        data=x_values,
        **kwargs
    ).set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)

    if filename:
        filepath = os.path.join(FIGURE_DIR, filename)
        plt.savefig(filepath)

    plt.show()
