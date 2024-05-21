import torch.nn as nn
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from skorch import NeuralNetClassifier
from torch import nn
from xgboost import XGBClassifier


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
    # "dt": DecisionTreeClassifier(),
    # "gauss": GaussianProcessClassifier(),
    # "gb": GradientBoostingClassifier(),
    # "mlp": MLPClassifier(),
    # "nb": GaussianNB(),
    # "net": NeuralNetClassifier(
    #     NeuralNetwork, max_epochs=10, lr=0.01, verbose=0, criterion=nn.BCELoss
    # ),
    # "rf": RandomForestClassifier(),
    "xgb": XGBClassifier(),
    "xgb_tuned": XGBClassifier(
        **{
            "booster": "gbtree",
            "lambda": 5.109935951174421e-07,
            "alpha": 0.06635237505310092,
            "subsample": 0.7404565076053088,
            "colsample_bytree": 0.6918787613388593,
            "eta": 0.09015511696579205,
            "gamma": 2.6131496243196327e-05,
            "max_depth": 5,
            "min_child_weight": 1,
            "grow_policy": "lossguide",
        }
    ),
}
