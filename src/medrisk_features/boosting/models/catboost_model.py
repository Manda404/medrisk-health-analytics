"""
CatBoost model wrapper for the medrisk boosting MLOps pipeline.

CatBoost natively handles categorical features without encoding,
which makes it particularly well-suited for clinical tabular data
where many variables are ordinal or nominal categories.

Installation: pip install medrisk-features[catboost]
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from medrisk_features.boosting.models.base import BaseBoostingModel


_CATBOOST_DEFAULTS: Dict[str, Any] = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "border_count": 128,
    "bagging_temperature": 1,
    "random_strength": 1,
    "random_seed": 42,
    "verbose": 0,
    "allow_writing_files": False,
}


class CatBoostModel(BaseBoostingModel):
    """
    CatBoost wrapper implementing BaseBoostingModel.

    Native categorical support: if cat_features is provided, CatBoost
    handles them directly without requiring OrdinalEncoder or OneHotEncoder.

    Parameters
    ----------
    params : dict or None
        CatBoost hyperparameters. Merged with _CATBOOST_DEFAULTS.
    task_type : str
        'binary_classification', 'multiclass_classification', or 'regression'.
    cat_features : list of str or None
        Column names of categorical features passed to CatBoost natively.
        When provided, ordinal encoding in the preprocessor is unnecessary.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "binary_classification",
        cat_features: Optional[List[str]] = None,
    ) -> None:
        merged = {**_CATBOOST_DEFAULTS, **(params or {})}
        super().__init__(params=merged)
        self.task_type = task_type
        self.cat_features = cat_features or []
        self._feature_names: list = []

    def _build_model(self) -> Any:
        """Instantiate the correct CatBoost estimator based on task type."""
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost is required. Install with: pip install medrisk-features[catboost]"
            ) from exc

        p = dict(self.params)

        if self.task_type == "regression":
            return CatBoostRegressor(**p)
        elif self.task_type == "multiclass_classification":
            return CatBoostClassifier(loss_function="MultiClass", **p)
        else:
            return CatBoostClassifier(loss_function="Logloss", **p)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> "CatBoostModel":
        self._feature_names = X_train.columns.tolist()
        self._model = self._build_model()

        # Resolve categorical feature indices for CatBoost
        cat_feature_indices = [
            X_train.columns.get_loc(c)
            for c in self.cat_features
            if c in X_train.columns
        ]

        fit_kwargs: Dict[str, Any] = {"cat_features": cat_feature_indices}
        if X_valid is not None and y_valid is not None:
            try:
                from catboost import Pool
                eval_set = Pool(X_valid, y_valid, cat_features=cat_feature_indices)
                fit_kwargs["eval_set"] = eval_set
            except ImportError:
                pass  # Already caught above

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self._model.predict(X).flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        proba = self._model.predict_proba(X)
        if self.task_type == "binary_classification" and proba.ndim == 2:
            return proba[:, 1]
        return proba

    def get_feature_importance(self) -> pd.Series:
        self._check_is_fitted()
        importance = self._model.get_feature_importance()
        return (
            pd.Series(importance, index=self._feature_names, name="importance")
            .sort_values(ascending=False)
        )

    def save(self, path: str) -> None:
        """Save model in CatBoost native cbm format."""
        self._check_is_fitted()
        if not path.endswith(".cbm"):
            path = path + ".cbm"
        self._model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "CatBoostModel":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "catboost is required. Install with: pip install medrisk-features[catboost]"
            ) from exc

        if not path.endswith(".cbm"):
            path = path + ".cbm"

        instance = cls.__new__(cls)
        BaseBoostingModel.__init__(instance)
        instance._model = CatBoostClassifier()
        instance._model.load_model(path)
        instance._is_fitted = True
        instance._feature_names = []
        instance.cat_features = []
        return instance
