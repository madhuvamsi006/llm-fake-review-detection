"""
XGBoost classifier for fake review detection.
Uses the classical TF-IDF + stylometric feature pipeline.
"""
import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

PARAM_GRID = {
    "n_estimators": [200, 300, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}


def train_xgboost(X_train, y_train, params: dict = None, tune: bool = False):
    if params is None:
        params = DEFAULT_PARAMS

    model = XGBClassifier(**params)

    if tune:
        grid = GridSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            PARAM_GRID,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_}")
        model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)

    return model


def save_model(model, path: str = "results/checkpoints/xgboost.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"XGBoost model saved to {path}")


def load_model(path: str = "results/checkpoints/xgboost.joblib"):
    return joblib.load(path)
