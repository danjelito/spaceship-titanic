import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier

from src import module

x_train, y_train, _ = module.load_dataset()
x_train, _ = module.data_preparation(x_train, _)
x_train = x_train.values.astype("float32")
y_train = y_train.values.astype("float32")


def objective(trial):
    param = {
        "verbosity": 0,
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }
    if param["booster"] in ["gbtree"]:
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["max_depth"] = trial.suggest_int("max_depth", 2, 9, step=1)
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

    model = XGBClassifier(**param)
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
    study.optimize(objective, n_trials=50)
    print()
    print(f"Best params: {study.best_params}")
