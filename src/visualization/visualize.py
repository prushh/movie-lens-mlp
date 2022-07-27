import os

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
# sns.set_palette('bright')


def show_class_distribution(df: pd.DataFrame):
    sns.set_theme(rc=custom_params)
    sns.set_palette('bright')
    distribution = (df.reset_index()
                    .groupby('rating_discrete')
                    .agg('count')
                    .reset_index()
                    .rename(columns={'index': 'count'}))

    # create pie chart
    plt.gca().axis("equal")

    pie = plt.pie(distribution['count'], startangle=0, pctdistance=0.9, radius=1.2)

    plt.title('Class distribution train set', fontsize=24)

    # Defining intervals labels

    labels = [f'{round(i, 2)}-{round(i + 0.45, 2)}' for i in np.arange(0.5, 5, 0.45)]
    plt.legend(pie[0],
               labels,
               bbox_to_anchor=(0.75, 0.5),
               loc="center right",
               fontsize=18,
               bbox_transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85)
    plt.show()
    plt.clf()
    plt.close()


def barplot_multiple_columns(groups: list, elements_group: list, data: list, title: str, filename: str = '',
                             save: bool = False, label_count: bool = False) -> None:
    sns.set_theme(rc=custom_params)

    sns.set_palette('bright')
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
        if label_count:
            ax.text(rs[idx], 1.05,
                    data[idx],
                    ha='center', va='bottom', rotation=90)

    ax.legend(fontsize=18, loc='lower left')
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

    if save:
        plt.savefig(filename)
    if print_plot:
        plt.show()
    else:
        plt.close()
