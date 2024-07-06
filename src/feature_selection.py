from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


def get_selected_features(x_train, y_train):

    estimator = LogisticRegression(max_iter=1000)
    selector = SelectFromModel(estimator=estimator, threshold="mean")
    selector = selector.fit(x_train, y_train)
    return selector.get_support()
