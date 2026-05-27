"""
Abstract base class for all boosting models.

All concrete model implementations (XGBoost, CatBoost, LightGBM) must
inherit from BaseBoostingModel and implement its interface.

This pattern ensures that:
  - the training pipeline is model-agnostic
  - swapping models requires zero changes in training/inference code
  - the BoostingModelFactory can instantiate any model transparently
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BaseBoostingModel(ABC):
    """
    Abstract interface for boosting models used in medrisk MLOps pipelines.

    All subclasses must expose the same public API so that the training
    pipeline, the pyfunc wrapper and the evaluator can work with any
    supported boosting library without modification.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters
        ----------
        params : dict or None
            Hyperparameters specific to the underlying library.
            Merged with sensible defaults defined in each subclass.
        """
        self.params: Dict[str, Any] = params or {}
        self._model: Any = None          # underlying library model object
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core interface (must be implemented by every subclass)
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> "BaseBoostingModel":
        """
        Train the model on (X_train, y_train).

        Parameters
        ----------
        X_train : pd.DataFrame  Training features.
        y_train : pd.Series     Training labels / targets.
        X_valid : pd.DataFrame  Optional validation features (for early stopping).
        y_valid : pd.Series     Optional validation labels.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return hard predictions (class labels or regression values).

        Parameters
        ----------
        X : pd.DataFrame  Input features.

        Returns
        -------
        np.ndarray  Shape (n_samples,).
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class probabilities.

        Parameters
        ----------
        X : pd.DataFrame  Input features.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_classes) for multiclass,
            or (n_samples,) of positive-class probability for binary.
        """

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """
        Return feature importance as a named Series sorted descending.

        Returns
        -------
        pd.Series  Index = feature names, values = importance scores.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Serialize the model to disk at the given path.

        Parameters
        ----------
        path : str  File path (without extension — subclass decides extension).
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseBoostingModel":
        """
        Deserialize a model from disk.

        Parameters
        ----------
        path : str  File path used in save().

        Returns
        -------
        BaseBoostingModel  Loaded model instance.
        """

    # ------------------------------------------------------------------
    # Shared helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if predict is called before fit."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. "
                "Call fit() before predict()."
            )

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.model_name}(params={self.params}, status={status})"
