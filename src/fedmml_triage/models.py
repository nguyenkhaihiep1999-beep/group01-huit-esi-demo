"""All three algorithms share the same train-fitted preprocessing contract."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import CATEGORICAL, CLASSES, CONTINUOUS, SEED, TEMPORAL


class TrainQuantileClipper(TransformerMixin, BaseEstimator):
    """Winsorize finite values with quantiles learned exclusively in fit."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Expected 0 <= lower < upper <= 1")
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.lower_ = np.full(self.n_features_in_, -np.inf)
        self.upper_ = np.full(self.n_features_in_, np.inf)
        for col in range(self.n_features_in_):
            finite = values[np.isfinite(values[:, col]), col]
            if finite.size:
                self.lower_[col], self.upper_[col] = np.quantile(finite, [self.lower, self.upper])
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected number of columns")
        return np.clip(values, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class FoldWeightedXGB(ClassifierMixin, BaseEstimator):
    """Map ESI 1..5 to XGBoost 0..4; compute sample weights on each fit's rows."""

    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.1, min_child_weight=1,
                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, weight_mode="none", n_jobs=2, random_state=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.weight_mode = weight_mode
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        if set(np.unique(y)) != set(CLASSES):
            raise ValueError("Every XGBoost training fold must contain all five ESI classes.")
        if self.weight_mode not in ("none", "balanced"):
            raise ValueError("Unsupported weight_mode")
        weights = compute_sample_weight("balanced", y) if self.weight_mode == "balanced" else None
        self.estimator_ = XGBClassifier(
            objective="multi:softprob", num_class=5, eval_metric="mlogloss", tree_method="hist",
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs, random_state=self.random_state, verbosity=0,
        )
        self.estimator_.fit(X, y - 1, sample_weight=weights)
        self.classes_ = np.asarray(CLASSES)
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_preprocessor():
    continuous = Pipeline([
        ("outlier", TrainQuantileClipper()),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    temporal = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("continuous", continuous, CONTINUOUS), ("temporal", temporal, TEMPORAL),
        ("categorical", categorical, CATEGORICAL),
    ], remainder="drop")


def model_spaces():
    return {
        "LogisticRegression": (
            LogisticRegression(solver="lbfgs", l1_ratio=0.0, max_iter=2000, random_state=SEED),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0], "classifier__class_weight": [None, "balanced"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, n_jobs=2),
            {"classifier__n_estimators": [150, 250], "classifier__max_depth": [None, 12, 20],
             "classifier__min_samples_leaf": [1, 3, 5], "classifier__max_features": ["sqrt", 0.8],
             "classifier__class_weight": [None, "balanced_subsample"]},
        ),
        "XGBoost": (
            FoldWeightedXGB(),
            {"classifier__n_estimators": [150, 300], "classifier__max_depth": [3, 5],
             "classifier__learning_rate": [0.05, 0.1], "classifier__min_child_weight": [1, 5],
             "classifier__subsample": [0.8, 1.0], "classifier__colsample_bytree": [0.8, 1.0],
             "classifier__reg_lambda": [1.0, 5.0], "classifier__weight_mode": ["none", "balanced"]},
        ),
    }
