import pickle
from random import randrange
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.metrics import zero_one_loss, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer

from src.models.config import param_grid_model
from src.utils.const import NUM_BINS


def balance(train_data: pd.DataFrame, train_target: pd.Series) -> Tuple:
    k_neighbors = np.min(train_target.value_counts()) - 1

    smt = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors),
                     tomek=TomekLinks(sampling_strategy='majority'))
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
        ('norm', norm)
    ])

    pipe.fit(train_data)
    train_data_proc = pipe.transform(train_data)
    test_data_proc = pipe.transform(test_data)

    return train_data_proc, test_data_proc


def fit_model(df: pd.DataFrame, model_group: str, easy_params: bool, model_to_test: str, test: bool):
    df = (df
          .assign(rating_discrete=pd.cut(df.loc[:, 'rating_mean'], bins=NUM_BINS, labels=False))
          .astype({'rating_discrete': 'int32'})
          .drop(columns=['rating_mean']))

    data = df.loc[:, df.columns != 'rating_discrete']
    target = df['rating_discrete']

    if not test:
        outer_results = []
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True)

        # TODO: now take only one model from a group, reduce also hyperparams
        if easy_params:
            random_conf = randrange(len(param_grid_model[model_group]))
            easy_param_grid = [param_grid_model[model_group][random_conf]]
            correct_param_grid = easy_param_grid
        else:
            correct_param_grid = param_grid_model[model_group]

        for model_name, estimator, param_grid in correct_param_grid:
            for fold, (train_idx, test_idx) in enumerate(cv_outer.split(data, y=target), 1):
                print(f'Fold {fold}')
                train_data, test_data = data.iloc[train_idx, :], data.iloc[test_idx, :]
                train_target, test_target = target[train_idx], target[test_idx]

                cv_inner = StratifiedKFold(n_splits=5, shuffle=True)

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
                y_pred = best_model.predict(test_data_proc)
                acc = accuracy_score(test_target, y_pred)
                loss = zero_one_loss(test_target, y_pred)
                outer_results.append(acc)

                if acc == max(outer_results):
                    filename = f'{model_name}.pkl'
                    pickle.dump(search.best_estimator_, open(filename, 'wb'))

                print(f'loss={loss:3f}, accuracy={acc:3f}, est={search.best_score_:3f}, cfg={search.best_params_}')

            print(f'[{model_name}] [test] Mean accuracy: {np.mean(outer_results):3f} - σ: {np.std(outer_results):3f}')
    else:
        # TODO: Use only 20% of dataset?
        test_eval(model_to_test, data, target)


def test_eval(filepath: str, test_data, test_target):
    estimator = pickle.load(open(filepath, 'rb'))
    y_pred = estimator.predict(test_data)
    acc = accuracy_score(test_target, y_pred)
    loss = zero_one_loss(test_target, y_pred)

    print(f'loss={loss:3f}, accuracy={acc:3f}')
