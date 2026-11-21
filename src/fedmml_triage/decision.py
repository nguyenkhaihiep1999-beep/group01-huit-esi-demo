"""Validation-only decision rules; probability estimates remain uncalibrated."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .config import CLASSES


def _validate_probabilities(probabilities):
    values = np.asarray(probabilities, dtype=float)
    if (values.ndim != 2 or values.shape[1] != len(CLASSES)
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Expected probability distributions in ESI order 1..5")
    return values


def weighted_prediction(probabilities, class_weights):
    values = _validate_probabilities(probabilities)
    weights = np.asarray(class_weights, dtype=float)
    if weights.shape != (5,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Expected five positive finite decision multipliers")
    return np.asarray(CLASSES)[np.argmax(values * weights, axis=1)]


def tune_class_weights(y_reference, probabilities, source="train_oof_cv"):
    """Bounded deterministic coordinate search, retaining argmax on score ties.

    Inputs should be out-of-fold CV predictions from train so that the holdout
    validation split remains completely untouched for unbiased model selection.
    This optimizes a decision rule, not calibrated probabilities.
    """
    values = _validate_probabilities(probabilities)
    y = np.asarray(y_reference)
    if y.shape != (len(values),) or set(np.unique(y)) != set(CLASSES):
        raise ValueError("Decision reference set must contain all five ESI classes")
    truth = y.astype(int) - 1

    def score(weights):
        prediction = np.argmax(values * weights, axis=1)
        cm = np.bincount(5 * truth + prediction, minlength=25).reshape(5, 5)
        return float(np.mean(2 * cm.diagonal() / (cm.sum(axis=0) + cm.sum(axis=1))))

    weights = np.ones(5)
    baseline = best = score(weights)
    grid = np.exp(np.linspace(np.log(0.5), np.log(2.0), 17))
    for _ in range(3):
        changed = False
        for column in range(5):
            candidates = []
            for multiplier in grid:
                trial = weights.copy()
                trial[column] = multiplier
                candidates.append((score(trial), trial))
            candidate_score, candidate = max(candidates, key=lambda item: item[0])
            if candidate_score > best + 1e-12:
                best, weights, changed = candidate_score, candidate, True
        if not changed:
            break
    return weights, {"source": source, "objective": "macro_f1",
                     "argmax_macro_f1": baseline, "tuned_macro_f1": best,
                     "class_order": list(CLASSES), "class_weights": weights.tolist(),
                     "multiplier_bounds": [0.5, 2.0], "probability_calibrated": False}


class ValidationDecisionClassifier(ClassifierMixin, BaseEstimator):
    """Attach a frozen argmax(p * alpha) rule to an already-fitted estimator."""

    def __init__(self, estimator, class_weights):
        self.estimator = estimator
        self.class_weights = class_weights

    @property
    def classes_(self):
        return self.estimator.classes_

    def fit(self, X, y):
        raise RuntimeError("Fit the base estimator on train, then tune decisions on validation")

    def __sklearn_is_fitted__(self):
        return hasattr(self.estimator, "classes_")

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return weighted_prediction(self.predict_proba(X), self.class_weights)


class FrozenSoftVoting(ClassifierMixin, BaseEstimator):
    """Soft voting of fitted RF/XGB pipelines, without refitting either model."""

    def __init__(self, random_forest, xgboost, rf_weight=0.5, class_weights=(1., 1., 1., 1., 1.)):
        self.random_forest = random_forest
        self.xgboost = xgboost
        self.rf_weight = rf_weight
        self.class_weights = class_weights

    @property
    def classes_(self):
        return np.asarray(CLASSES)

    def fit(self, X, y):
        raise RuntimeError("Soft voting requires two fitted training-only pipelines")

    def __sklearn_is_fitted__(self):
        return all(hasattr(model, "classes_") for model in [self.random_forest, self.xgboost])

    def predict_proba(self, X):
        if not 0 < self.rf_weight < 1:
            raise ValueError("Soft voting requires 0 < rf_weight < 1")
        for model in [self.random_forest, self.xgboost]:
            if not np.array_equal(model.classes_, CLASSES):
                raise ValueError("Unexpected ESI probability column order")
        return (self.rf_weight * self.random_forest.predict_proba(X)
                + (1 - self.rf_weight) * self.xgboost.predict_proba(X))

    def predict(self, X):
        return weighted_prediction(self.predict_proba(X), self.class_weights)
