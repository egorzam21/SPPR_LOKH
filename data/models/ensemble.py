from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
from config.params import RANDOM_SEED

def create_final_model():
    return {
        'rf': RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'gbm': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            random_state=RANDOM_SEED
        )
    }

def train_ensemble(models, X, y):
    for m in models.values():
        m.fit(X, y)
    return models

def ensemble_predict(models, X):
    preds = []
    for name, model in models.items():
        preds.append(model.predict(X))

    weights = {'rf': 0.65, 'gbm': 0.35}
    return sum(weights[n] * p for p, n in zip(preds, models.keys()))
