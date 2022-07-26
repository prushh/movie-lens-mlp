import os
import pickle
from random import randrange
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import zero_one_loss, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer

from src.models.config import param_grid_model, best_param_grid_model
from src.utils.const import NUM_BINS, MODEL_RESULTS_CSV, MODEL_RESULTS_DIR
from src.utils.util_models import add_row_to_df_sk
from src.visualization.visualize import plot_roc


def balance(train_data: pd.DataFrame, train_target: pd.Series) -> Tuple:
    k_neighbors = np.min(train_target.value_counts()) - 1

    smt = SMOTE(k_neighbors=k_neighbors)
    train_data_smt, train_target_smt = smt.fit_resample(train_data, train_target)

    return train_data_smt, train_target_smt


def preprocess(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple:
    features = [
        'year',
        'title_length',
        'runtime',
        'rating_count',
        'tag_count'
    ]

    scaler = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('minmax', MinMaxScaler(), features)
        ])

    norm = Normalizer(norm='l2')

    pipe = Pipeline(steps=[
        ('scaler', scaler),
        # ('norm', norm)
    ])

    pipe.fit(train_data)
    train_data_proc = pipe.transform(train_data)
    test_data_proc = pipe.transform(test_data)

    return train_data_proc, test_data_proc


def fit_model(df: pd.DataFrame, model_group: str, easy_params: bool, best_conf: bool, roc: bool):
    df = (df
          .assign(rating_discrete=pd.cut(df.loc[:, 'rating_mean'], bins=NUM_BINS, labels=False))
          .astype({'rating_discrete': 'int32'})
          .drop(columns=['rating_mean']))

    data = df.loc[:, df.columns != 'rating_discrete']
    target = df['rating_discrete']
    df_results = pd.DataFrame()

    N_SPLITS = 5

    cv_outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

    if easy_params:
        random_conf = randrange(len(param_grid_model[model_group]))
        easy_param_grid = [param_grid_model[model_group][random_conf]]
        get_param_grid = easy_param_grid[0][2]
        tmp_dict = {}
        for key in get_param_grid.keys():
            rnd_idx = randrange(len(get_param_grid[key]))
            tmp_dict[key] = [get_param_grid[key][rnd_idx]]
        correct_param_grid = [(easy_param_grid[0][0], easy_param_grid[0][1], tmp_dict)]
    elif best_conf:
        correct_param_grid = best_param_grid_model[model_group]
    else:
        correct_param_grid = param_grid_model[model_group]

    for model_name, estimator, param_grid in correct_param_grid:
        print(f'Model name: {model_name}')
        outer_results = []
        outer_f1_results = []
        for fold, (train_idx, test_idx) in enumerate(cv_outer.split(data, y=target), 1):
            print(f'Fold {fold}')
            train_data, test_data = data.iloc[train_idx, :], data.iloc[test_idx, :]
            train_target, test_target = target[train_idx], target[test_idx]

            cv_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

            train_data_smt, train_target_smt = balance(train_data, train_target)
            train_data_proc, test_data_proc = preprocess(train_data_smt, test_data)

            search = GridSearchCV(estimator=estimator,
                                  param_grid=param_grid,
                                  scoring='accuracy',
                                  cv=cv_inner,
                                  refit=True,
                                  n_jobs=-1,
                                  verbose=3)

            search.fit(train_data_proc, train_target_smt)

            best_model = search.best_estimator_
            save_fold_model(fold, model_name, best_model)

            acc, loss, f1_test = test_eval(fold, model_name, test_data_proc, test_target, roc)
            outer_results.append(acc)
            outer_f1_results.append(f1_test)
            df_results = add_row_to_df_sk(model_name, fold, df_results, loss, acc, f1_test,
                                          search.best_params_)
            if not os.path.exists(MODEL_RESULTS_DIR):
                os.mkdir(MODEL_RESULTS_DIR)
            if not os.path.exists(MODEL_RESULTS_CSV):
                os.mkdir(MODEL_RESULTS_CSV)
            df_results.to_csv(os.path.join(MODEL_RESULTS_CSV, f'out_{model_group}.csv'), encoding='utf-8')

            if acc == max(outer_results):
                filename = f'{model_name}.pkl'
                filepath = os.path.join(MODEL_RESULTS_DIR, filename)
                pickle.dump(search.best_estimator_, open(filepath, 'wb'))

            print(f'loss={loss:3f}, acc={acc:3f} ,f1-score={f1_test:3f}, cfg={search.best_params_}')

        print(
            f'[{model_name}] [test] Mean accuracy: {np.mean(outer_results):3f} - Mean f1-score: {np.mean(outer_f1_results):3f}')


def test_eval(fold: int, model_name: str, test_data: pd.DataFrame, test_target: pd.DataFrame, roc: bool) -> Tuple:
    filename = f'{fold}_{model_name}.pkl'
    filepath = os.path.join(MODEL_RESULTS_DIR, model_name, filename)
    est = pickle.load(open(filepath, 'rb'))

    is_tree, is_svm = False, False
    if model_name == 'decision_tree_classifier':
        is_tree = True
    elif model_name == 'svc':
        is_svm = True
    if is_svm:
        p = np.array(est.decision_function(test_data))
        y_pred_prob = np.exp(p) / np.sum(np.exp(p), axis=1, keepdims=True)
    else:
        y_pred_prob = est.predict_proba(test_data)

    y_pred = est.predict(test_data)
    acc = accuracy_score(test_target, y_pred)
    loss = zero_one_loss(test_target, y_pred)
    f1_test = f1_score(test_target, y_pred, average='weighted')

    if not is_tree and roc:
        plot_roc(test_target, y_pred_prob, model_name)

    return acc, loss, f1_test


def save_fold_model(fold: int, model_name: str, best_est, notebook: bool = False) -> None:
    filename = f'{fold}_{model_name}.pkl'
    if notebook:
        fold_model_results_dir = os.path.join('..', MODEL_RESULTS_DIR, model_name)
    else:
        fold_model_results_dir = os.path.join(MODEL_RESULTS_DIR, model_name)
    if not os.path.exists(fold_model_results_dir):
        os.mkdir(fold_model_results_dir)

    filepath = os.path.join(fold_model_results_dir, filename)
    pickle.dump(best_est, open(filepath, 'wb'))
