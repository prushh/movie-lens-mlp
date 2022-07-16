import os

import matplotlib.pyplot
import numpy as np
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


def barplot_multiple_columns(groups: list, elements_group: list, data: list, title: str, filename: str = '',
                             save: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))

    # X deve essere il range del numero di gruppi di grafico
    X = np.arange(len(groups))
    width = 0.1
    rs = [X]
    # deve scorrere il range del numero di barre che ci sono nel gruppo
    for idx in range(1, len(elements_group)):
        tmp = rs[idx - 1]
        rs.append(
            [val + width for val in tmp]
        )
    # va creata un'array che contiene un elemento per gruppo di grafico quindi in questo caso
    # l'elemento avrà cardinalità |len(df['model_name'].unique())|
    for idx, elm in enumerate(elements_group):
        # df[df['balance']==elm][scores].to_numpy().squeeze() = [f1_random, f1_decision, f1_gaussian, f1_quadratic]
        ax.bar(x=rs[idx], height=data[idx], label=elm, width=width)

    ax.legend(loc='lower left')
    loc_ticks = [(val + (len(elements_group) / 2) * width) - width / 2 for val in
                 range(len(groups))]
    upper_labels = [val.upper() for val in groups]
    ax.set_title(title, fontsize=24)
    ax.set_xticks(loc_ticks)
    ax.set_xticklabels(upper_labels)

    if save:
        plt.savefig(filename)

    plt.show()


def histplot(x_values: pd.Series, title: str, xlabel: str, ylabel: str, filename: str = '', save: bool = False,
             **kwargs) -> None:
    sns.set_theme(rc=custom_params)
    sns.histplot(
        data=x_values,
        **kwargs
    ).set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)

    # TODO: think about saving plot in different way (specific function?)
    if save:
        filepath = os.path.join(FIGURE_DIR, filename)
        plt.savefig(filepath)

    plt.show()


def kdeplot(x_values: pd.Series, title: str, xlabel: str, ylabel: str, filename: str = '', save: bool = False,
            print_plot=True,
            **kwargs) -> None:
    sns.set_theme(rc=custom_params)
    sns.kdeplot(
        data=x_values,
        **kwargs
    ).set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)

    # TODO: think about saving plot in different way (specific function?)
    if save:
        plt.savefig(filename)
    if print_plot:
        plt.show()
    else:
        plt.close()
