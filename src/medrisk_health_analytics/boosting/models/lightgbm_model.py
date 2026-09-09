"""
LightGBM model wrapper for the medrisk boosting MLOps pipeline.

LightGBM is included as an optional third boosting option.
It is generally faster than XGBoost on large datasets and handles
categorical features natively (similar to CatBoost).

Installation: pip install medrisk-health-analytics[lightgbm]
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from medrisk_health_analytics.boosting.models.base import BaseBoostingModel

_LIGHTGBM_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


class LightGBMModel(BaseBoostingModel):
    """
    LightGBM wrapper implementing BaseBoostingModel.

    Parameters
    ----------
    params : dict or None
        LightGBM hyperparameters. Merged with _LIGHTGBM_DEFAULTS.
    task_type : str
        'binary_classification' or 'multiclass_classification'.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "binary_classification",
    ) -> None:
        merged = {**_LIGHTGBM_DEFAULTS, **(params or {})}
        super().__init__(params=merged)
        self.task_type = task_type
        self._feature_names: list = []

    def _build_model(self) -> Any:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required. Install with: pip install medrisk-health-analytics[lightgbm]"
            ) from exc

        p = dict(self.params)

        if self.task_type == "multiclass_classification":
            return lgb.LGBMClassifier(objective="multiclass", **p)
        if self.task_type == "binary_classification":
            return lgb.LGBMClassifier(objective="binary", **p)

        raise ValueError(
            f"Unsupported task_type '{self.task_type}'. "
            "Supported values: binary_classification, multiclass_classification."
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> LightGBMModel:
        self._feature_names = X_train.columns.tolist()
        self._model = self._build_model()

        fit_kwargs: Dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["callbacks"] = []  # suppress verbose output

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        proba = self._model.predict_proba(X)
        if self.task_type == "binary_classification" and proba.ndim == 2:
            return proba[:, 1]
        return proba

    def get_feature_importance(self) -> pd.Series:
        self._check_is_fitted()
        importance = self._model.feature_importances_
        return pd.Series(importance, index=self._feature_names, name="importance").sort_values(
            ascending=False
        )

    def save(self, path: str) -> None:
        """Save model in LightGBM text format via the underlying booster."""
        self._check_is_fitted()
        if not path.endswith(".txt"):
            path = path + ".txt"
        self._model.booster_.save_model(path)

    @classmethod
    def load(cls, path: str, task_type: str = "binary_classification") -> LightGBMModel:
        """
        Deserialize a model from disk.

        Parameters
        ----------
        path : str
            File path used in save() (without extension).
        task_type : str
            Task type used during training. Defaults to 'binary_classification'.

        Notes
        -----
        Bug #1 fix: task_type is now set on the loaded instance so predict_proba()
        can correctly slice probabilities for binary classification.
        Bug #2 fix: we reconstruct a LGBMClassifier (which has predict_proba) by
        loading the saved booster into it, rather than returning a raw lgb.Booster
        which lacks predict_proba().
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required. Install with: pip install medrisk-health-analytics[lightgbm]"
            ) from exc

        if not path.endswith(".txt"):
            path = path + ".txt"

        instance = cls.__new__(cls)
        BaseBoostingModel.__init__(instance)
        instance.task_type = task_type  # Bug #1 fix: required by predict_proba()

        # Bug #2 fix: load into LGBMClassifier (has predict_proba) instead of bare Booster
        classifier = lgb.LGBMClassifier()
        classifier._Booster = lgb.Booster(model_file=path)  # type: ignore[attr-defined]
        instance._model = classifier
        instance._is_fitted = True
        instance._feature_names = []
        return instance
