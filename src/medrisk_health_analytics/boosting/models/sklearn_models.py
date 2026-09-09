"""Uniform wrappers for scikit-learn classification models."""

from __future__ import annotations

import pickle
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from medrisk_health_analytics.boosting.models.base import BaseBoostingModel


class SklearnClassifierModel(BaseBoostingModel):
    """Shared implementation for scikit-learn classifier wrappers."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "binary_classification",
    ) -> None:
        super().__init__(params=params)
        self.task_type = task_type
        self._feature_names: list[str] = []

    @abstractmethod
    def _build_model(self) -> Any:
        """Create the configured scikit-learn estimator."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> SklearnClassifierModel:
        del X_valid, y_valid
        self._feature_names = X_train.columns.tolist()
        self._model = self._build_model()
        self._model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(self._model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        probabilities = np.asarray(self._model.predict_proba(X))
        if self.task_type == "binary_classification":
            return probabilities[:, 1]
        return probabilities

    def get_feature_importance(self) -> pd.Series:
        self._check_is_fitted()
        if hasattr(self._model, "feature_importances_"):
            values = np.asarray(self._model.feature_importances_)
        elif hasattr(self._model, "coef_"):
            coefficients = np.asarray(self._model.coef_)
            values = np.abs(coefficients).mean(axis=0)
        else:
            raise RuntimeError(f"{self.model_name} does not expose feature importance")
        return pd.Series(values, index=self._feature_names, name="importance").sort_values(
            ascending=False
        )

    def save(self, path: str) -> None:
        """Serialize the fitted estimator as a pickle artifact."""
        self._check_is_fitted()
        output_path = path if path.endswith(".pkl") else f"{path}.pkl"
        with open(output_path, "wb") as stream:
            pickle.dump(self._model, stream)

    @classmethod
    def load(cls, path: str) -> SklearnClassifierModel:
        """Load a fitted estimator saved by this wrapper."""
        input_path = path if path.endswith(".pkl") else f"{path}.pkl"
        with open(input_path, "rb") as stream:
            estimator = pickle.load(stream)
        instance = cls()
        instance._model = estimator
        instance._is_fitted = True
        instance._feature_names = list(getattr(estimator, "feature_names_in_", []))
        return instance


class LogisticRegressionModel(SklearnClassifierModel):
    """Wrapper around scikit-learn logistic regression."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "binary_classification",
    ) -> None:
        defaults = {"max_iter": 1000, "random_state": 42}
        super().__init__({**defaults, **(params or {})}, task_type=task_type)

    def _build_model(self) -> Any:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**self.params)


class RandomForestModel(SklearnClassifierModel):
    """Wrapper around scikit-learn random forest."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "binary_classification",
    ) -> None:
        defaults = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
        super().__init__({**defaults, **(params or {})}, task_type=task_type)

    def _build_model(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**self.params)


class GradientBoostingModel(SklearnClassifierModel):
    """Wrapper around scikit-learn gradient boosting."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "binary_classification",
    ) -> None:
        defaults = {"n_estimators": 200, "learning_rate": 0.05, "random_state": 42}
        super().__init__({**defaults, **(params or {})}, task_type=task_type)

    def _build_model(self) -> Any:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(**self.params)


__all__ = [
    "GradientBoostingModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "SklearnClassifierModel",
]
