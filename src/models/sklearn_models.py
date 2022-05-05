from typing import Tuple, Dict

import numpy as np
import pandas as pd
import sklearn
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

from src.models.config import param_grid_tree, param_grid_forest, param_grid_boosting, param_grid_svc, param_grid_nb
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


def tuning_hyper_parameters(train_data: np.ndarray, train_target: np.ndarray, estimator: sklearn.base.BaseEstimator,
                            param_grid: Dict):
    '''
    search = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          cv=5,
                          verbose=3,
                          scoring='accuracy',
                          n_jobs=-1)

    search.fit(train_data, train_target)
    predicted_target = search.predict(test_data)
    loss = zero_one_loss(test_target, predicted_target)
    score = search.score(test_data, test_target)

    print(f'Loss: {loss:.3f}')
    print(f'Score: {score:.3f}')
    '''


def fit_model(df: pd.DataFrame, model_name: str):
    df = (df
          .assign(rating_discrete=pd.cut(df.loc[:, 'rating_mean'], bins=NUM_BINS, labels=False))
          .astype({'rating_discrete': 'int32'})
          .drop(columns=['rating_mean']))

    data = df.loc[:, df.columns != 'rating_discrete']
    target = df['rating_discrete']

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2,
                                                                        stratify=df['rating_discrete'])
    train_data, test_data = preprocess(train_data, test_data)

    counts = np.bincount(train_target)
    class_weight = dict(enumerate(1. / counts))

    param_grid_model = {
        'tree_based': [
            (DecisionTreeClassifier(class_weight=class_weight), param_grid_tree),
            (RandomForestClassifier(class_weight=class_weight), param_grid_forest),
            (GradientBoostingClassifier(), param_grid_boosting)
        ],
        'svm': [
            (SVC(), param_grid_svc)
        ],
        'naive_bayes': [
            (GaussianNB(), param_grid_nb),
            (QuadraticDiscriminantAnalysis(), param_grid_nb)
        ]
    }

    for estimator, param_grid in param_grid_model[model_name]:
        print(f'Estimator: {estimator}')
        search = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              cv=5,
                              verbose=3,
                              scoring='accuracy',
                              n_jobs=-1)

        search.fit(train_data, train_target)
        predicted_target = search.predict(test_data)
        loss = zero_one_loss(test_target, predicted_target)
        score = search.score(test_data, test_target)

        print(f'Loss: {loss:.3f}')
        print(f'Score: {score:.3f}')
