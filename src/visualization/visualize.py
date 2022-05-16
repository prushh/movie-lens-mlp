import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.utils.const import FIGURE_DIR

custom_params = {
    'figure.figsize': (16, 8),
    'lines.linewidth': 3,
    'axes.titlesize': 20,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}
# sns.set_theme(rc=custom_params)


def histplot(x_values: pd.Series, title: str, xlabel: str, ylabel: str, filename: str = '', **kwargs) -> None:
    sns.set_theme(rc=custom_params)
    sns.histplot(
        data=x_values,
        **kwargs
    ).set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)

    # TODO: think about saving plot in different way (specific function?)
    if filename:
        filepath = os.path.join(FIGURE_DIR, filename)
        plt.savefig(filepath)

    plt.show()
