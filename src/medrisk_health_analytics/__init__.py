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

__version__ = "0.2.0"
__author__ = "Rostand Surel"
__email__ = "s239150.eps@gmail.com"
__license__ = "MIT"

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
    # Exceptions (for external catch blocks)
    "MedRiskError",
    "SchemaValidationError",
    "FeatureEngineeringError",
    "MissingRequiredColumnError",
    "InvalidConfigurationError",
    # Metadata
    "__version__",
]
