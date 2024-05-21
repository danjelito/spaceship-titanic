import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src import module, feature_selection

# load dataset
x_train, y_train, _ = module.load_dataset()

# data preparation
x_train, _ = module.data_preparation(x_train, _)
x_train = x_train.values.astype("float32")
y_train = y_train.values.astype("float32").ravel()

# feature selection
mask = feature_selection.get_selected_features(x_train, y_train)
x_train = x_train[:, mask]


def objective(trial):

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 1, 12),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 0.1, 10),
        "objective": trial.suggest_categorical(
            "objective", ["Logloss", "CrossEntropy"]
        ),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["Ordered", "Plain"]
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "4gb",
        "verbose": False,
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    model = CatBoostClassifier(**param)
    y_pred = cross_val_predict(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=3,
    )
    acc = accuracy_score(y_train, y_pred)
    return acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print()
    print(f"Best params: {study.best_params}")
