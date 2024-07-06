import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from skorch import NeuralNetClassifier
from torch import nn
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LazyLinear(out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


models = {
    # "ada": AdaBoostClassifier(),
    # "catboost": CatBoostClassifier(verbose=False),
    # "dt": DecisionTreeClassifier(),
    # "gb": GradientBoostingClassifier(),
    # "knn": KNeighborsClassifier(),
    # "lda": LinearDiscriminantAnalysis(),
    # "lgb": LGBMClassifier(),
    # "log_reg": LogisticRegression(),
    # "mlp": MLPClassifier(),
    # "nb": GaussianNB(),
    "net": NeuralNetClassifier(
        NeuralNetwork, max_epochs=10, lr=0.01, verbose=0, criterion=nn.BCELoss
    ),
    # "qda": QuadraticDiscriminantAnalysis(),
    # "rf": RandomForestClassifier(),
    # "xgb": XGBClassifier(),
    # "xgb_tuned": XGBClassifier(
    #     **{
    #         "booster": "gbtree",
    #         "lambda": 5.109935951174421e-07,
    #         "alpha": 0.06635237505310092,
    #         "subsample": 0.7404565076053088,
    #         "colsample_bytree": 0.6918787613388593,
    #         "eta": 0.09015511696579205,
    #         "gamma": 2.6131496243196327e-05,
    #         "max_depth": 5,
    #         "min_child_weight": 1,
    #         "grow_policy": "lossguide",
    #     }
    # ),
    # "catboost_tuned": CatBoostClassifier(
    #     **{
    #         "verbose": False,
    #         "n_estimators": 441,
    #         "learning_rate": 0.08624571897240016,
    #         "depth": 12,
    #         "l2_leaf_reg": 1.7275826170857882,
    #         "objective": "Logloss",
    #         "colsample_bylevel": 0.07940489532774267,
    #         "boosting_type": "Ordered",
    #         "bootstrap_type": "Bernoulli",
    #         "subsample": 0.9170590299364719,
    #     }
    # ),
}
