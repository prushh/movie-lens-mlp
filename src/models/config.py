import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# params with different batch size and batch norm
# param_layers = {
#     'input_act': [nn.LeakyReLU()],
#     'hidden_act': [nn.LeakyReLU()],
#     'hidden_size': [512],
#     'num_hidden_layers': [3],
#     'dropout': [0.2],
#     'batch_norm': [False, True],
#     'output_fn': [None]
# }
#
# param_grid_mlp = {
#     'num_epochs': [200],
#     'starting_lr': [1e-3],
#     'batch_size': [16, 32, 64, 512, 2048, 16384],
#     'optim': [torch.optim.Adam],
#     'momentum': [0.9],
#     'weight_decay': [1e-7]
# }

param_layers = {
    'input_act': [nn.LeakyReLU()],
    'hidden_act': [nn.LeakyReLU()],
    'hidden_size': [512],
    'num_hidden_layers': [3],
    'dropout': [0.2],
    'batch_norm': [True],
    'output_fn': [None]
}

param_grid_mlp = {
    'num_epochs': [200],
    'starting_lr': [1e-3],
    'batch_size': [2 ** i for i in range(3, 15)],
    'optim': [torch.optim.Adam],
    'momentum': [0.9],
    'weight_decay': [1e-5]
}

# TEST CONFIGURATION
# param_layers = {
#     'input_act': [nn.ReLU()],
#     'hidden_act': [nn.LeakyReLU()],
#     'hidden_size': [512],
#     'num_hidden_layers': [5],
#     'dropout': [0.2],
#     'batch_norm': [False],
#     'output_fn': [None]
# }
#
# param_grid_mlp = {
#     'num_epochs': [150],
#     'starting_lr': [1e-3],
#     'batch_size': [128, 256],
#     'optim': [torch.optim.Adam],
#     'momentum': [0.9],
#     'weight_decay': [0.000001]
# }


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
    'svm': [
        ('svc', SVC(), param_grid_svc)
    ],
    'naive_bayes': [
        ('gaussian_nb', GaussianNB(), param_grid_nb),
        ('qda', QuadraticDiscriminantAnalysis(), param_grid_qda)
    ]
}
