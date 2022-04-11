import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams.update({'figure.figsize': (16, 10), 'figure.dpi': 100, 'font.size': 18})


def plot_distribution(x_values: pd.Series, title: str, x_label: str, y_label: str, bins: int = 25):
    # Plot instance
    sns.histplot(
        data=x_values,
        bins=bins,
        kde=True,
        line_kws={"linewidth": 3})
    plt.gca().set(title=title, ylabel=y_label, xlabel=x_label)
    plt.show()
