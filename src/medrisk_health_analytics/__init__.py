"""
medrisk-health-analytics
================
Production-ready Python package for medical, metabolic and lifestyle
feature engineering in healthcare machine learning.

Quick start
-----------
>>> import pandas as pd
>>> from medrisk_health_analytics import FeatureEngineeringPipeline
>>>
>>> df = pd.read_csv("patient_data.csv")
>>> pipeline = FeatureEngineeringPipeline(validate_schema=True)
>>> df_enriched = pipeline.transform(df)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("medrisk-health-analytics")
except PackageNotFoundError:
    __version__ = "0.4.1"
__author__ = "Rostand Surel"
__email__ = "s239150.eps@gmail.com"
__license__ = "MIT"

from medrisk_health_analytics.api import (
    DatasetAnalysis,
    DatasetAnalyzer,
    DatasetLoader,
    DatasetPreprocessor,
    ModelPredictor,
    ModelTrainer,
    analyze_dataset,
    build_features,
    load_dataset,
    predict,
    preprocess_dataset,
    train_model,
)
from medrisk_health_analytics.boosting.models import (
    CatBoostModel,
    GradientBoostingModel,
    LightGBMModel,
    LogisticRegressionModel,
    ModelFactory,
    RandomForestModel,
    XGBoostModel,
)
from medrisk_health_analytics.evaluation import EvaluationReport, ModelEvaluator
from medrisk_health_analytics.monitoring import (
    DataDriftMonitor,
    DataQualityReport,
    DataQualityValidator,
    DriftReport,
)
from medrisk_health_analytics.pipeline import FeatureEngineeringPipeline
from medrisk_health_analytics.utils.exceptions import (
    FeatureEngineeringError,
    InvalidConfigurationError,
    MedRiskError,
    MissingRequiredColumnError,
    SchemaValidationError,
)

__all__ = [
    # Main entry point
    "FeatureEngineeringPipeline",
    "DataDriftMonitor",
    "DataQualityReport",
    "DataQualityValidator",
    "DriftReport",
    "DatasetAnalysis",
    "DatasetLoader",
    "DatasetAnalyzer",
    "DatasetPreprocessor",
    "ModelTrainer",
    "ModelPredictor",
    "ModelEvaluator",
    "EvaluationReport",
    "ModelFactory",
    "XGBoostModel",
    "CatBoostModel",
    "LightGBMModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "load_dataset",
    "analyze_dataset",
    "preprocess_dataset",
    "build_features",
    "train_model",
    "predict",
    # Exceptions (for external catch blocks)
    "MedRiskError",
    "SchemaValidationError",
    "FeatureEngineeringError",
    "MissingRequiredColumnError",
    "InvalidConfigurationError",
    # Metadata
    "__version__",
]
