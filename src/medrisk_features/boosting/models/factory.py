"""
BoostingModelFactory — instantiate any supported boosting model by name.

This factory pattern decouples the training pipeline from specific
library imports. Adding a new model type requires only:
  1. Implementing the BaseBoostingModel interface
  2. Registering the class in REGISTRY below

Usage
-----
>>> model = BoostingModelFactory.create("xgboost", params={"max_depth": 4})
>>> model.fit(X_train, y_train)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from medrisk_features.boosting.models.base import BaseBoostingModel


# ---------------------------------------------------------------------------
# Registry — maps model_type string → implementation class
# (imported lazily to keep library imports optional)
# ---------------------------------------------------------------------------

def _get_registry() -> Dict[str, Type[BaseBoostingModel]]:
    """Build and return the model registry with lazy imports."""
    from medrisk_features.boosting.models.xgboost_model import XGBoostModel
    from medrisk_features.boosting.models.catboost_model import CatBoostModel
    from medrisk_features.boosting.models.lightgbm_model import LightGBMModel

    return {
        "xgboost":  XGBoostModel,
        "catboost": CatBoostModel,
        "lightgbm": LightGBMModel,
    }


SUPPORTED_MODELS = ["xgboost", "catboost", "lightgbm"]


class BoostingModelFactory:
    """
    Factory that creates a BaseBoostingModel instance by model_type name.

    Example
    -------
    >>> model = BoostingModelFactory.create(
    ...     model_type="xgboost",
    ...     params={"max_depth": 4, "n_estimators": 200},
    ...     task_type="binary_classification",
    ... )
    >>> model.fit(X_train, y_train)
    """

    @staticmethod
    def create(
        model_type: str,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "binary_classification",
        **kwargs: Any,
    ) -> BaseBoostingModel:
        """
        Instantiate a boosting model by type name.

        Parameters
        ----------
        model_type : str
            One of: 'xgboost', 'catboost', 'lightgbm'.
        params : dict or None
            Hyperparameters for the chosen model. If None, defaults are used.
        task_type : str
            ML task: 'binary_classification', 'multiclass_classification',
            or 'regression'.
        **kwargs
            Additional keyword arguments forwarded to the model constructor
            (e.g. cat_features for CatBoostModel).

        Returns
        -------
        BaseBoostingModel
            A configured (but not yet fitted) model instance.

        Raises
        ------
        ValueError
            If model_type is not in the supported model registry.
        """
        registry = _get_registry()
        model_type_lower = model_type.strip().lower()

        if model_type_lower not in registry:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported models: {sorted(registry.keys())}"
            )

        model_class = registry[model_type_lower]
        return model_class(params=params, task_type=task_type, **kwargs)

    @staticmethod
    def list_supported() -> list:
        """Return the list of currently supported model types."""
        return list(_get_registry().keys())
