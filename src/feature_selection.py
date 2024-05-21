from sklearn.feature_selection import (
    RFE,
    RFECV,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFromModel,
    chi2,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_selected_features(x_train, y_train):

    # estimator = DecisionTreeClassifier()
    # selector = RFE(step=1, n_features_to_select=0.5, estimator=estimator)
    # selector = selector.fit(x_train, y_train)
    # return selector.support_

    # transformer = SelectFdr(alpha=0.01)
    # transformer.fit(x_train, y_train)
    # return transformer.get_support()

    # return mutual_info_classif(x_train, y_train) > 0.1

    # estimator = DecisionTreeClassifier()
    estimator = LogisticRegression(max_iter=1000)
    selector = SelectFromModel(estimator=estimator, threshold="mean")
    selector = selector.fit(x_train, y_train)
    return selector.get_support()
