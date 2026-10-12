"""Fit-only resampling and fold-local class weighting for the three model families."""

import hashlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from ..config import CLASSES, SEED
from ..models import FoldWeightedXGB


STRATEGIES = (
    "baseline", "balanced_global", "balanced_45_soft", "balanced_5_boost",
    "oversample_5_to_4", "undersample_4_to_5",
)


def class_counts(y):
    return {str(c): int(np.count_nonzero(np.asarray(y) == c)) for c in CLASSES}


def prepare_training(y, strategy, random_state=SEED):
    """Return indices into these fit rows only, plus optional normalized weights."""
    y = np.asarray(y, dtype=int)
    if strategy not in STRATEGIES:
        raise ValueError("Unknown balancing strategy")
    if set(np.unique(y)) != set(CLASSES):
        raise ValueError("Training rows must contain all five ESI classes")
    indices = np.arange(len(y))
    weights = None
    rng = np.random.default_rng(random_state)
    if strategy == "oversample_5_to_4":
        rows5 = np.flatnonzero(y == 5)
        extra = max(0, int(np.count_nonzero(y == 4)) - len(rows5))
        indices = np.concatenate([indices, rng.choice(rows5, size=extra, replace=True)])
        rng.shuffle(indices)
    elif strategy == "undersample_4_to_5":
        rows4 = np.flatnonzero(y == 4)
        keep = min(len(rows4), int(np.count_nonzero(y == 5)))
        indices = np.concatenate([indices[y != 4], rng.choice(rows4, size=keep, replace=False)])
        rng.shuffle(indices)
    elif strategy != "baseline":
        counts = class_counts(y)
        class_weights = {c: len(y) / (len(CLASSES) * counts[str(c)]) for c in CLASSES}
        if strategy == "balanced_45_soft":
            class_weights[4] *= 0.85
            class_weights[5] *= 0.5
        elif strategy == "balanced_5_boost":
            class_weights[5] *= 2.0
        weights = np.asarray([class_weights[int(c)] for c in y], dtype=float)
        weights /= weights.mean()
    audit = {
        "strategy": strategy, "fit_input_rows": len(y), "fit_output_rows": len(indices),
        "class_counts_before": class_counts(y), "class_counts_after": class_counts(y[indices]),
        "unique_source_rows": len(np.unique(indices)),
        "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "indices_within_fit_rows": bool(np.all((indices >= 0) & (indices < len(y)))),
        "sample_weight_mean": None if weights is None else float(weights.mean()),
        "sample_weight_min": None if weights is None else float(weights.min()),
        "sample_weight_max": None if weights is None else float(weights.max()),
        "additional_weighting_with_sampling": False,
    }
    return indices, weights, audit


class BalancedTrainingClassifier(ClassifierMixin, BaseEstimator):
    """Resample only in fit, after the preprocessing fitted on original fold rows."""

    def __init__(self, estimator, strategy="baseline", random_state=SEED):
        self.estimator = estimator
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        indices, weights, self.fit_audit_ = prepare_training(y, self.strategy, self.random_state)
        self.sample_indices_ = indices
        self.classes_ = np.asarray(CLASSES)
        self.n_features_in_ = X.shape[1]
        self.label_offset_ = 0
        if self.strategy == "baseline":
            self.estimator_ = clone(self.estimator)
        elif isinstance(self.estimator, FoldWeightedXGB):
            # FoldWeightedXGB does not accept externally provided sample_weight.
            # Clone its underlying XGBClassifier with the same frozen hyperparameters.
            from xgboost import XGBClassifier
            parameters = self.estimator.get_params()
            parameters.pop("weight_mode")
            self.estimator_ = XGBClassifier(
                **parameters, objective="multi:softprob", num_class=5,
                eval_metric="mlogloss", tree_method="hist", verbosity=0,
            )
            self.label_offset_ = 1
        else:
            self.estimator_ = clone(self.estimator).set_params(class_weight=None)
        self.estimator_.fit(np.asarray(X)[indices], y[indices] - self.label_offset_,
                            **({"sample_weight": weights} if weights is not None else {}))
        expected = self.classes_ - self.label_offset_
        if not np.array_equal(self.estimator_.classes_, expected):
            raise ValueError("Unexpected ESI probability column order")
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
