"""
medrisk_features.boosting
=========================
End-to-end MLOps layer for boosting models (XGBoost, CatBoost, LightGBM)
on Databricks with full MLflow integration.

Public API
----------
>>> from medrisk_features.boosting import (
...     TrainingConfig,
...     train_boosting_model,
...     predict_with_registered_model,
...     BoostingPyFuncModel,
...     BoostingModelFactory,
... )

Quick example
-------------
>>> config = TrainingConfig(
...     target_column="TARGET",
...     id_columns=["client_id", "pers_id"],
...     model_type="xgboost",
...     experiment_name="/Shared/experiments/boosting",
...     registered_model_name="workspace.schema.boosting_model",
... )
>>> result = train_boosting_model(df, config=config)
>>> print(result.metrics["test_roc_auc"])

>>> predictions = predict_with_registered_model(
...     model_uri="models:/workspace.schema.boosting_model@Champion",
...     input_df=df_new,
... )
"""

from medrisk_features.boosting.config.schemas import (
    TrainingConfig,
    InferenceConfig,
    TrainingResult,
    SplitStrategy,
    TaskType,
)
from medrisk_features.boosting.models.base import BaseBoostingModel
from medrisk_features.boosting.models.factory import BoostingModelFactory
from medrisk_features.boosting.pyfunc.boosting_pyfunc_model import BoostingPyFuncModel
from medrisk_features.boosting.training.trainer import train_boosting_model
from medrisk_features.boosting.inference.predictor import predict_with_registered_model
from medrisk_features.boosting.preprocessing.tabular_preprocessor import TabularPreprocessor
from medrisk_features.boosting.utils.spark_utils import to_pandas, to_spark

__all__ = [
    # Configuration
    "TrainingConfig",
    "InferenceConfig",
    "TrainingResult",
    "SplitStrategy",
    "TaskType",
    # Models
    "BaseBoostingModel",
    "BoostingModelFactory",
    # Pyfunc (core)
    "BoostingPyFuncModel",
    # Training & inference
    "train_boosting_model",
    "predict_with_registered_model",
    # Preprocessing
    "TabularPreprocessor",
    # Spark utils
    "to_pandas",
    "to_spark",
]
