import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

def select_important_features(X, y, k=25):
    if X.shape[1] <= k:
        return X, list(X.columns)

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    cols = X.columns[selector.get_support()].tolist()

    return X[cols], cols
