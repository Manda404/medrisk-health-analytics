"""
MedRiskPyFuncModel — mlflow.pyfunc.PythonModel wrapper for medrisk-features.

This module wraps the FeatureEngineeringPipeline inside a standard
mlflow.pyfunc.PythonModel so that the entire feature engineering logic
can be:
  - logged to an MLflow experiment as a versioned artifact
  - registered in the MLflow Model Registry (including Unity Catalog)
  - loaded back for batch inference on Databricks
  - served via MLflow Model Serving

The model is intentionally kept as a pure feature transformer:
it receives raw patient data and returns an enriched DataFrame
ready for a downstream ML model (XGBoost, CatBoost, etc.).

Usage (Databricks notebook)
---------------------------
>>> from medrisk_features.mlflow import log_pipeline, load_pipeline
>>>
>>> # Log after fitting
>>> model_info = log_pipeline(
...     pipeline=pipeline,
...     experiment_name="/Shared/experiments/medrisk",
...     registered_model_name="workspace.my_schema.medrisk_pipeline",
... )
>>>
>>> # Load and run inference
>>> loaded = mlflow.pyfunc.load_model(model_info.model_uri)
>>> df_enriched = loaded.predict(df_raw)
"""

from __future__ import annotations

import json
import os
import pickle
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

if TYPE_CHECKING:
    pass


class MedRiskPyFuncModel:
    """
    MLflow PythonModel wrapper for the medrisk-features pipeline.

    This class encapsulates the full FeatureEngineeringPipeline inside
    the mlflow.pyfunc contract so it can be serialized, versioned,
    registered, and loaded back for inference on any environment.

    Artifacts stored alongside the model
    -------------------------------------
    - pipeline.pkl     : serialized FeatureEngineeringPipeline instance
    - config.json      : pipeline configuration (age_group_strategy, etc.)
    - feature_names.json: list of output feature column names (logged after fit)

    load_context / predict contract
    --------------------------------
    load_context(context):
        Deserializes the pipeline from context.artifacts["pipeline"].

    predict(context, model_input):
        Accepts a pandas DataFrame of raw patient data and returns
        the fully enriched DataFrame produced by the pipeline.
    """

    # Keys expected in context.artifacts
    ARTIFACT_PIPELINE = "pipeline"
    ARTIFACT_CONFIG = "config"
    ARTIFACT_FEATURE_NAMES = "feature_names"

    def load_context(self, context: Any) -> None:
        """
        Load all artifacts from the MLflow model context.

        Called automatically by MLflow when the model is loaded via
        mlflow.pyfunc.load_model(). Deserializes the pipeline and config.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            Provides access to artifact paths via context.artifacts.
        """
        # -- Load the serialized pipeline ---------------------------------
        pipeline_path = context.artifacts[self.ARTIFACT_PIPELINE]
        with open(pipeline_path, "rb") as f:
            self.pipeline = pickle.load(f)

        # -- Load the configuration ---------------------------------------
        config_path = context.artifacts[self.ARTIFACT_CONFIG]
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        # -- Load feature names (optional — may not exist on older models) --
        feature_names_path = context.artifacts.get(self.ARTIFACT_FEATURE_NAMES)
        if feature_names_path and os.path.exists(feature_names_path):
            with open(feature_names_path, encoding="utf-8") as f:
                self.feature_names: Optional[list] = json.load(f)
        else:
            self.feature_names = None

    def predict(self, context: Any, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """
        Apply the feature engineering pipeline to raw input data.

        Called automatically by MLflow when predictions are requested,
        e.g. via loaded_model.predict(df) or during batch inference.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            MLflow model context (not used directly in predict, but
            required by the PythonModel interface).
        model_input : pd.DataFrame
            Raw patient data DataFrame. Must contain at minimum:
            - Age
            - glucose_fasting
            - bmi
        params : dict or None
            Optional inference-time parameters (required by mlflow >= 2.6
            PythonModel interface). Currently unused.

        Returns
        -------
        pd.DataFrame
            Fully enriched DataFrame with all engineered features.

        Raises
        ------
        ValueError
            If model_input is not a pandas DataFrame.
        """
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError(
                f"model_input must be a pandas DataFrame, got {type(model_input).__name__}."
            )

        # Apply the full feature engineering pipeline
        df_enriched = self.pipeline.transform(model_input)
        return df_enriched


def _get_pyfunc_class():
    """
    Lazily import mlflow.pyfunc.PythonModel to keep mlflow optional.
    Returns the MedRiskPyFuncModel class with PythonModel as base,
    or a plain version if mlflow is not installed.
    """
    try:
        import mlflow.pyfunc

        class _MedRiskPyFuncModelWithBase(MedRiskPyFuncModel, mlflow.pyfunc.PythonModel):
            """
            Production class: MedRiskPyFuncModel with mlflow.pyfunc.PythonModel
            as base class. Used when mlflow is available.
            """

            pass

        return _MedRiskPyFuncModelWithBase
    except ImportError:
        return MedRiskPyFuncModel
