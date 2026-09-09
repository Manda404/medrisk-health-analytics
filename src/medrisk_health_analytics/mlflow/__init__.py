"""
medrisk_health_analytics.mlflow
=======================
Optional MLflow integration for the medrisk-health-analytics pipeline.

This subpackage provides:
  - MedRiskPyFuncModel : mlflow.pyfunc.PythonModel wrapper for the pipeline
  - log_pipeline       : log a pipeline to an MLflow experiment
  - load_pipeline      : load a logged pipeline from the Model Registry
  - set_model_alias    : assign a Unity Catalog alias (e.g. 'Champion')

Requires mlflow to be installed:
  pip install medrisk-health-analytics[mlflow]
  # or: pip install mlflow

Usage
-----
>>> from medrisk_health_analytics.mlflow import log_pipeline, load_pipeline
>>>
>>> result = log_pipeline(
...     pipeline=pipeline,
...     experiment_name="/Shared/experiments/medrisk",
...     df_sample=df.head(10),
...     registered_model_name="workspace.schema.medrisk_pipeline",
... )
>>> loaded = load_pipeline(result.model_uri)
>>> df_enriched = loaded.predict(df_raw)
"""

from medrisk_health_analytics.mlflow.pyfunc_model import MedRiskPyFuncModel
from medrisk_health_analytics.mlflow.tracker import (
    PipelineLogResult,
    load_pipeline,
    log_pipeline,
    set_model_alias,
)

__all__ = [
    "MedRiskPyFuncModel",
    "PipelineLogResult",
    "log_pipeline",
    "load_pipeline",
    "set_model_alias",
]
