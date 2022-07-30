import itertools
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from src.models.config import param_layers, param_grid_mlp


def mu_confidence_interval(data: np.ndarray) -> Dict:
    """
    Compute mean and t_student with 90% of accuracy
    :param data: the ndarray that contains metric data
    :return: Dict with metric values
    """
    t = 2.13
    mu = np.mean(data)
    standard_deviation = np.std(data)
    M = data.shape[0]
    t_student = t * standard_deviation / np.sqrt(M)
    first_interval = mu - t_student
    second_interval = mu + t_student
    return {
        'mu': mu,
        't_student': t_student,
        'first_interval': first_interval,
        'second_interval': second_interval
    }


def get_best_configuration_mlp(cfg: int, p_layer, p_grid_mlp) -> Tuple:
    """
    Compute cartesian product using mlp hyperparams and return
    the configuration specified by the index cfg
    :param cfg: the configuration index to retrieve
    :param p_layer: the network architecture parameters
    :param p_grid_mlp: the network hyperparams
    :return: Tuple with best configuration
    """
    hyper_parameters_model_all = itertools.product(
        p_layer['input_act'],
        p_layer['hidden_act'],
        p_layer['hidden_size'],
        p_layer['num_hidden_layers'],
        p_layer['dropout'],
        p_layer['batch_norm'],
        p_layer['output_fn'],
        p_grid_mlp['num_epochs'],
        p_grid_mlp['starting_lr'],
        p_grid_mlp['batch_size'],
        p_grid_mlp['optim'],
        p_grid_mlp['momentum'],
        p_grid_mlp['weight_decay'],
    )
    return list(hyper_parameters_model_all)[cfg]


def summary_statistics_model(df_score: pd.DataFrame, dict_: Dict, model: str, train: bool = False,
                             scikit: bool = False) -> pd.DataFrame:
    """
    Well formatted print with metrics, configurations, etc. about the specified model.
    Create a sample with the printed metrics and concat it to the df_score input DataFrame
    :param df_score: the DataFrame that will contain a new sample
    :param dict_: the Dict with the metrics to be printed
    :param model: the model name
    :param train: the flag that specifies which metric are used
    :param scikit: the flag to specify model types
    :return: the updated DataFrame with the new sample
    """
    if not train:
        print(
            f"Best configuration {model} mean metrics:\n"
            f"f1_score: {dict_['f1']['mu']} ±{dict_['f1']['t_student']}\n"
            f"loss: {dict_['loss']['mu']} ±{dict_['loss']['t_student']}\n"
            f"acc: {dict_['acc']['mu']} ±{dict_['acc']['t_student']}\n\n"
            f"Best hyperparams configuration:"
        )
        if model == "mlp" or model == 'MLP':
            best_cfg_mlp_all = get_best_configuration_mlp(int(dict_['conf']), param_layers, param_grid_mlp)
            for idx, key in enumerate(param_layers.keys()):
                print(f"{key}: {best_cfg_mlp_all[idx]}")
            for idx, key in enumerate(param_grid_mlp.keys(), 7):
                print(f"{key}: {best_cfg_mlp_all[idx]}")
        else:
            print(f"{dict_['conf']}")

        new_test_score = pd.DataFrame({
            'model': [model],
            'f1_mu': [dict_['f1']['mu']],
            'acc_mu': [dict_['acc']['mu']],
            'loss_mu': [dict_['loss']['mu']],
            'f1_ci': [dict_['f1']['t_student']],
            'acc_ci': [dict_['acc']['t_student']],
            'loss_ci': [dict_['loss']['t_student']],
        })
        df_score = pd.concat([df_score, new_test_score], ignore_index=True)
    else:
        if scikit:
            print(
                f"Best configuration {model} mean metrics:\n"
                f"train f1: {dict_['mean_f1_train']['mu']} ±{dict_['mean_f1_train']['t_student']}\n"
                f"validation f1: {dict_['mean_f1_val']['mu']} ±{dict_['mean_f1_val']['t_student']}\n"
            )
            new_test_score = pd.DataFrame({
                'model': [model],
                'train_score': [dict_['mean_f1_train']['mu']],
                'val_score': [dict_['mean_f1_val']['mu']],
                'train_ci': [dict_['mean_f1_train']['t_student']],
                'val_ci': [dict_['mean_f1_val']['t_student']]
            })
        else:
            print(
                f"Best configuration {model} mean metrics:\n"
                f"train f1: {dict_['train_score']['mu']} ±{dict_['train_score']['t_student']}\n"
                f"validation f1: {dict_['val_score']['mu']} ±{dict_['val_score']['t_student']}\n"
            )
            new_test_score = pd.DataFrame({
                'model': [model],
                'train_score': [dict_['train_score']['mu']],
                'val_score': [dict_['val_score']['mu']],
                'train_ci': [dict_['train_score']['t_student']],
                'val_ci': [dict_['val_score']['t_student']]
            })
        df_score = pd.concat([df_score, new_test_score], ignore_index=True)
    return df_score


def find_max_f1_cfg(df: pd.DataFrame, scikit: bool = False) -> np.ndarray:
    """
    Find the best configuration based on mean val f1-score
    :param df: the DataFrame that contains model outputs
    :param scikit: the flag to specify model types
    :return: ndarray with the best configuration indices
    """
    col_names = ('mean_test_score', 'params') if scikit else ('mean_f1_val', 'cfg')

    cfg = []
    for fold in df['fold'].unique():
        idx = df[df['fold'] == fold][col_names[0]].idxmax()
        cfg.append(df.iloc[idx][col_names[1]])
    cfgs = np.unique(np.array(cfg))
    return cfgs


def find_best_conf(lst_conf, df: pd.DataFrame, scikit: bool = False) -> Dict:
    """
    Create Dict with statistics for all metrics, based on the different configurations inside lst_conf
    :param lst_conf: the list that contains the configuration indexes
    :param df: the DataFrame that contains the metrics
    :param scikit: the flag to specify model types
    :return: Dict with the best statistics based on mean val
    """
    conf = []
    for idx, cfg in enumerate(lst_conf):
        if scikit:
            one_sample = {
                'mean_f1_val': mu_confidence_interval(df[df['params'] == cfg]['mean_test_score']),
                'mean_f1_train': mu_confidence_interval(df[df['params'] == cfg]['mean_train_score'])
            }
        else:
            one_sample = {
                'f1': mu_confidence_interval(df[df['cfg'] == cfg]['f1_test']),
                'loss': mu_confidence_interval(df[df['cfg'] == cfg]['loss_test']),
                'acc': mu_confidence_interval(df[df['cfg'] == cfg]['acc_test']),
                'mean_f1_val': mu_confidence_interval(df[df['cfg'] == cfg]['mean_f1_val'])
            }
        conf.append(one_sample)
        conf[idx]['conf'] = cfg
        if not scikit:
            conf[idx]['acc']['mu'] /= 100
            conf[idx]['acc']['t_student'] /= 100
    max_val = conf[0]
    for elm in conf:
        if max_val['mean_f1_val']['mu'] < elm['mean_f1_val']['mu'] and max_val['mean_f1_val']['t_student'] > \
                elm['mean_f1_val']['t_student']:
            max_val = elm
    return max_val


def calculate_statistics_sklearn(df: pd.DataFrame, model: str) -> Dict:
    """
    Create a well-formatted Dict that contains the metrics
    :param df: the DataFrame that contains metrics
    :param model: the model name
    :return: well-formatted Dict
    """
    res = {'f1': mu_confidence_interval(df[df['model'] == model]['f1_test']),
           'loss': mu_confidence_interval(df[df['model'] == model]['loss_test']),
           'acc': mu_confidence_interval(df[df['model'] == model]['acc_test']),
           'conf': df[df['model'] == model]['cfg'].unique()}
    return res


def calculate_statistics_sklearn_train(df: pd.DataFrame, first: bool = True) -> Dict:
    """
    Retrieve sample of some model using index slicing
    :param df: the DataFrame that contains data
    :param first: the flag to get different samples
    :return: well-formatted Dict with metrics
    """
    if first:
        res = {'train_score': mu_confidence_interval(df['mean_train_score'].iloc[0:5]),
               'val_score': mu_confidence_interval(df['mean_test_score'].iloc[0:5]),
               }
    else:
        res = {'train_score': mu_confidence_interval(df['mean_train_score'].iloc[5:10]),
               'val_score': mu_confidence_interval(df['mean_test_score'].iloc[5:10]),
               }
    return res


def calculate_statistics_mlp_train(cfg: int, df: pd.DataFrame) -> Dict:
    """
    Create a well-formatted Dict that contains the metrics
    :param cfg: the configuration index
    :param df: the DataFrame that contains metric
    :return: well-formatted Dict
    """
    res = {'train_score': mu_confidence_interval(df[df['cfg'] == cfg]['mean_f1_train']),
           'val_score': mu_confidence_interval(df[df['cfg'] == cfg]['mean_f1_val']),
           }
    return res


def add_value_array(old_list: List, df_test: pd.DataFrame, col_name: str, last_idx: int) -> None:
    """
    Add values by ref to old_list, useful to plot multi bar graph correctly
    :param old_list: the list that is updated with new values
    :param df_test: the DataFrame that contains test results
    :param col_name: it specifies if we want metrics or confidence intervals
    :param last_idx: the index useful to retrieve metrics and confidence intervals columns
    :return: None
    """
    for model_name in df_test['model'].unique():
        if col_name == 'metrics':
            metric_value = df_test[df_test['model'] == model_name].iloc[:, 1:last_idx].iloc[0]
        else:
            metric_value = df_test[df_test['model'] == model_name].iloc[:, last_idx:].iloc[0]

        old_list.append(np.array(metric_value))
