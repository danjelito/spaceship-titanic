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
            "lambda": 8.999047800441808e-05,
            "alpha": 0.5510484808227769,
            "subsample": 0.2462366922641045,
            "colsample_bytree": 0.9856817861224756,
            "eta": 0.04151338214066351,
            "gamma": 0.0007209896246520909,
            "max_depth": 6,
            "min_child_weight": 4,
            "grow_policy": "depthwise",
        }
    ),
}
