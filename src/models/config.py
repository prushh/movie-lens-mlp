import numpy as np
import torch.nn as nn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

parameters = {
    'input_act': [nn.ReLU(), nn.LeakyReLU(), nn.Tanh()],
    'hidden_act': [nn.ReLU(), nn.LeakyReLU()],
    'dropout': np.arange(0, 1.1, .1),
    'batch_norm': [False, True],
    'output_fn': [None, nn.Softmax()]
}

param_grid_svc = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

param_grid_nb = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

param_grid_qda = {
    'reg_param': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'tol': [0.0001, 0.001, 0.01, 0.1],
}

param_grid_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(5, 20, 5)
}

param_grid_forest = {
    'n_estimators': np.arange(100, 1000, 200),
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4]
}

param_grid_boosting = {
    'n_estimators': np.arange(100, 1000, 200),
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': np.arange(5, 20, 5),
    'learning_rate': [0.01, 0.5, 1.0]
}

param_grid_model = {
    'tree_based': [
        ('random_forest_classifier', RandomForestClassifier(), param_grid_forest),
        ('decision_tree_classifier', DecisionTreeClassifier(), param_grid_tree),
        ('gradient_boosting_classifier', GradientBoostingClassifier(), param_grid_boosting)
    ],
    'test': [
        ('decision_tree_classifier', DecisionTreeClassifier(), {
            'criterion': ['gini'],
            'max_depth': [None]
        })
    ],
    'svm': [
        ('svc', SVC(), param_grid_svc)
    ],
    'naive_bayes': [
        ('gaussian_nb', GaussianNB(), param_grid_nb),
        ('qda', QuadraticDiscriminantAnalysis(), param_grid_qda)
    ]
}
