"""
medrisk_health_analytics.boosting
=========================
End-to-end MLOps layer for boosting models (XGBoost, CatBoost, LightGBM)
on Databricks with full MLflow integration.

Public API
----------
>>> from medrisk_health_analytics.boosting import (
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

from medrisk_health_analytics.boosting.config.schemas import (
    InferenceConfig,
    SplitStrategy,
    TaskType,
    TrainingConfig,
    TrainingResult,
)
from medrisk_health_analytics.boosting.data.dataset_splitter import (
    DatasetSplitResult,
    SplitTablesResult,
    create_train_test_tables,
    split_dataframe,
)
from medrisk_health_analytics.boosting.explainability.shap_explainer import (
    BoostingShapExplainer,
    ShapResult,
)
from medrisk_health_analytics.boosting.inference.predictor import predict_with_registered_model
from medrisk_health_analytics.boosting.models import (
    CatBoostModel,
    GradientBoostingModel,
    LightGBMModel,
    LogisticRegressionModel,
    ModelFactory,
    RandomForestModel,
    XGBoostModel,
)
from medrisk_health_analytics.boosting.models.base import BaseBoostingModel
from medrisk_health_analytics.boosting.models.factory import BoostingModelFactory
from medrisk_health_analytics.boosting.preprocessing.tabular_preprocessor import TabularPreprocessor
from medrisk_health_analytics.boosting.pyfunc.boosting_pyfunc_model import BoostingPyFuncModel
from medrisk_health_analytics.boosting.training.holdout import (
    HoldoutEvaluationResult,
    compute_binary_holdout_metrics,
    evaluate_registered_model_on_holdout,
)
from medrisk_health_analytics.boosting.training.trainer import train_boosting_model
from medrisk_health_analytics.boosting.utils.spark_utils import to_pandas, to_spark
from medrisk_health_analytics.evaluation import EvaluationReport, ModelEvaluator

__all__ = [
    # Configuration
    "TrainingConfig",
    "InferenceConfig",
    "TrainingResult",
    "SplitStrategy",
    "TaskType",
    # Dataset preparation
    "DatasetSplitResult",
    "SplitTablesResult",
    "split_dataframe",
    "create_train_test_tables",
    # Models
    "BaseBoostingModel",
    "BoostingModelFactory",
    "ModelFactory",
    "XGBoostModel",
    "CatBoostModel",
    "LightGBMModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    # Pyfunc (core)
    "BoostingPyFuncModel",
    # Training & inference
    "train_boosting_model",
    "HoldoutEvaluationResult",
    "compute_binary_holdout_metrics",
    "evaluate_registered_model_on_holdout",
    "predict_with_registered_model",
    # Preprocessing
    "TabularPreprocessor",
    # Spark utils
    "to_pandas",
    "to_spark",
    # Explainability (SHAP)
    "BoostingShapExplainer",
    "ShapResult",
    "ModelEvaluator",
    "EvaluationReport",
]
