"""
XGBoost model wrapper for the medrisk boosting MLOps pipeline.

Wraps xgboost.XGBClassifier behind the BaseBoostingModel
interface so the rest of the pipeline never calls xgboost directly.

Installation: pip install medrisk-health-analytics[xgboost]
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from medrisk_features.boosting.models.base import BaseBoostingModel

# Default hyperparameters optimised for tabular binary classification
_XGBOOST_DEFAULTS: Dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


class XGBoostModel(BaseBoostingModel):
    """
    XGBoost wrapper implementing BaseBoostingModel.

    Supports early stopping via the validation set and switches objectives
    for binary or multiclass classification based on task_type.

    Parameters
    ----------
    params : dict or None
        XGBoost hyperparameters. Merged with _XGBOOST_DEFAULTS.
    task_type : str
        'binary_classification' or 'multiclass_classification'.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "binary_classification",
    ) -> None:
        merged = {**_XGBOOST_DEFAULTS, **(params or {})}
        super().__init__(params=merged)
        self.task_type = task_type
        self._feature_names: list = []

    def _build_model(self) -> Any:
        """Instantiate the correct XGBoost estimator based on task type."""
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required. Install with: pip install medrisk-health-analytics[xgboost]"
            ) from exc

        p = {k: v for k, v in self.params.items() if k != "use_label_encoder"}

        if self.task_type == "multiclass_classification":
            return xgb.XGBClassifier(objective="multi:softprob", **p)
        if self.task_type == "binary_classification":
            return xgb.XGBClassifier(objective="binary:logistic", **p)

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
    ) -> XGBoostModel:
        self._feature_names = X_train.columns.tolist()
        self._model = self._build_model()

        fit_kwargs: Dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["verbose"] = False

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        proba = self._model.predict_proba(X)
        # For binary classification return probability of positive class
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
        """Save model as XGBoost native JSON format."""
        self._check_is_fitted()
        if not path.endswith(".json"):
            path = path + ".json"
        self._model.save_model(path)

    @classmethod
    def load(cls, path: str, task_type: str = "binary_classification") -> "XGBoostModel":
        """
        Deserialize a model from disk.

        Parameters
        ----------
        path : str
            File path used in save() (without extension).
        task_type : str
            Task type used during training. Defaults to 'binary_classification'.
            Pass 'multiclass_classification' if the model was trained for multiclass.
        """
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required. Install with: pip install medrisk-health-analytics[xgboost]"
            ) from exc

        if not path.endswith(".json"):
            path = path + ".json"

        instance = cls.__new__(cls)
        BaseBoostingModel.__init__(instance)
        instance.task_type = task_type  # Bug #1 fix: task_type required by predict_proba()
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(path)
        instance._is_fitted = True
        instance._feature_names = []
        return instance
