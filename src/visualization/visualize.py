import os
from itertools import cycle

import matplotlib.pyplot
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.utils.const import FIGURE_DIR, NUM_BINS

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


def plot_roc(y_test, y_pred_proba, model_name):
    # Metric ROC AUC
    classes = [i for i in range(y_pred_proba.shape[1])]
    y_test = label_binarize(y_test, classes=classes)

    n_class = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    fig, ax = plt.subplots()
    for i in range(n_class):
        ax.plot(fpr[i], tpr[i], label=f'ROC curve class {i} (area = {round(roc_auc[i],2)})')
    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver operating characteristic model {model_name}')
    ax.legend(loc="lower right")
    plt.show()


def plot_roc_multiclass(y_test, y_pred_proba, is_training=False):
    """
    This function returns ROC AUC score calculated with a 'one-vs-rest'
    approach. If called during training and validation it returns only
    ROC AUC score, else, if called during performance measure stage, it
    will also plots roc curves for each class.
    Code adapted from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """

    classes = [i for i in range(y_pred_proba.shape[1])]

    # binarize: [2] -> [0,0,1,0,0,0,0,0,0]
    y_test = label_binarize(y_test, classes=classes)
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        # roc_auc_score(y_test, y_pred_proba, multi_class='ovr') is the same as sum all auc (fpr[i], tpr[i])
        # and divide them for n_classes
        roc_auc[i] = auc(fpr[i], tpr[i])

    # first aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # ROC AUC score calculated during training, no plots needed
    if is_training:
        return roc_auc["macro"]

    # Plot all ROC curves at test time
    colors = cycle(['red',
                    'slategrey',
                    'firebrick',
                    'aqua',
                    'olive',
                    'darkorange',
                    'darkviolet',
                    'cornflowerblue',
                    'darkslategrey',
                    'gold'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic multiclass ')
    plt.legend(loc='lower right')
    #plt.savefig(join(plots_dir, f'roc_curve_{model_type}{fold}'))
    #plt.clf()
    plt.show()

    return roc_auc["macro"]
