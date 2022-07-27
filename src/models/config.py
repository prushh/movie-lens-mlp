import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

param_layers = {
    'input_act': [nn.LeakyReLU()],
    'hidden_act': [nn.LeakyReLU()],
    'hidden_size': [512],
    'num_hidden_layers': [3, 5],
    'dropout': [0.2, 0.3, 0.5],
    'batch_norm': [False, True],
    'output_fn': [None]
}

param_grid_mlp = {
    'num_epochs': [200],
    'starting_lr': [1e-3],
    'batch_size': [128, 256],
    'optim': [torch.optim.Adam, torch.optim.SGD],
    'momentum': [0.6, 0.9],
    'weight_decay': [1e-5, 1e-7]
}

param_grid_svc = {
    'C': [1, 10, 100],
    'gamma': [1, 0.1, 0.01],
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

param_grid_model = {
    'tree_based': [
        ('random_forest_classifier', RandomForestClassifier(), param_grid_forest),
        ('decision_tree_classifier', DecisionTreeClassifier(), param_grid_tree)
    ],
    'svm': [
        ('svc', SVC(), param_grid_svc)
    ],
    'naive_bayes': [
        ('gaussian_nb', GaussianNB(), param_grid_nb),
        ('qda', QuadraticDiscriminantAnalysis(), param_grid_qda)
    ]
}

best_param_grid_model = {
    'tree_based': [
        ('random_forest_classifier', RandomForestClassifier(),
         {'n_estimators': [700], 'max_features': ['sqrt'], 'max_depth': [4]}),
        ('decision_tree_classifier', DecisionTreeClassifier(), {'criterion': ['entropy'], 'max_depth': [15]})
    ],
    'svm': [
        ('svc', SVC(), {'C': [100], 'gamma': [0.01], 'kernel': ['rbf']})
    ],
    'naive_bayes': [
        ('gaussian_nb', GaussianNB(), {'var_smoothing': [8.111308307896872e-07]}),
        ('qda', QuadraticDiscriminantAnalysis(), {'reg_param': [0.001], 'tol': [0.0001]})
    ]
}

best_param_layers = {
    'input_act': [nn.LeakyReLU()],
    'hidden_act': [nn.LeakyReLU()],
    'hidden_size': [512],
    'num_hidden_layers': [3],
    'dropout': [0.2],
    'batch_norm': [True],
    'output_fn': [None]
}

best_param_grid_mlp = {
    'num_epochs': [50],
    'starting_lr': [1e-3],
    'batch_size': [128],
    'optim': [torch.optim.Adam],
    'momentum': [0.9],
    'weight_decay': [1e-5]
}

param_layers_batch = {
    'input_act': [nn.LeakyReLU()],
    'hidden_act': [nn.LeakyReLU()],
    'hidden_size': [512],
    'num_hidden_layers': [3],
    'dropout': [0.2],
    'batch_norm': [True, False],
    'output_fn': [None]
}

param_grid_mlp_batch = {
    'num_epochs': [200],
    'starting_lr': [1e-3],
    'batch_size': [16, 32, 64, 512, 2048, 16384],
    'optim': [torch.optim.Adam],
    'momentum': [0.9],
    'weight_decay': [1e-5]
}
