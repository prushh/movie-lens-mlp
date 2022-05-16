import pickle
from collections import Counter
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.models.config import param_grid_tree, param_grid_forest, param_grid_boosting, param_grid_svc, param_grid_nb, \
    param_grid_qda
from src.utils.const import NUM_BINS


def preprocess(train_data: np.ndarray, test_data: np.ndarray) -> Tuple:
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


def fit_model(df: pd.DataFrame, model_name: str, refit_model: bool = False):
    df = (df
          .assign(rating_discrete=pd.cut(df.loc[:, 'rating_mean'], bins=NUM_BINS, labels=False))
          .astype({'rating_discrete': 'int32'})
          .drop(columns=['rating_mean']))

    data = df.loc[:, df.columns != 'rating_discrete']
    target = df['rating_discrete']

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2,
                                                                        stratify=df['rating_discrete'])

    train_data_proc, test_data_proc = preprocess(train_data, test_data)

    train_target, test_target = train_target.to_numpy(), test_target.to_numpy()

    smt_tom = SMOTETomek(smote=SMOTE(k_neighbors=4), tomek=TomekLinks(sampling_strategy='majority'))
    train_data_smt_tom, train_target_smt_tom = smt_tom.fit_resample(train_data_proc, train_target)

    param_grid_model = {
        'tree_based': [
            (RandomForestClassifier(), param_grid_forest),
            (DecisionTreeClassifier(), param_grid_tree),
            (GradientBoostingClassifier(), param_grid_boosting)
        ],
        'test': [
            (DecisionTreeClassifier(), {
                'criterion': ['gini'],
                'max_depth': [None]
            })
        ],
        'svm': [
            (SVC(), param_grid_svc)
        ],
        'naive_bayes': [
            (GaussianNB(), param_grid_nb),
            (QuadraticDiscriminantAnalysis(), param_grid_qda)
        ]
    }

    if refit_model:
        print('REFITTING')
        for idx, (estimator, param_grid) in enumerate(param_grid_model[model_name]):
            print(f'Estimator: {estimator}')
            search = GridSearchCV(estimator=estimator,
                                  param_grid=param_grid,
                                  cv=5,
                                  verbose=3,
                                  scoring='accuracy',
                                  n_jobs=-1)

            search.fit(train_data_smt_tom, train_target_smt_tom)
            predicted_target = search.predict(test_data_proc)
            loss = zero_one_loss(test_target, predicted_target)
            score = search.score(test_data_proc, test_target)

            print(f'Best val_score: {search.best_score_}')
            print(f'Loss: {loss:.3f}')
            print(f'Roc_Auc: {score:.3f}')

            filename = f'{idx}_{model_name}.pkl'
            pickle.dump(search, open(filename, 'wb'))
    else:
        print('RELOAD MODEL')
        for idx, (estimator, _) in enumerate(param_grid_model[model_name]):
            filename = f'{idx}_{model_name}.pkl'
            search = pickle.load(open(filename, 'rb'))
            predicted_target = search.predict(test_data_proc)
            loss = zero_one_loss(test_target, predicted_target)
            score = search.score(test_data_proc, test_target)

            print(f'Loss: {loss:.3f}')
            print(f'Roc_Auc: {score:.3f}')
