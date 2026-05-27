"""
medrisk_features.mlflow
=======================
Optional MLflow integration for the medrisk-features pipeline.

This subpackage provides:
  - MedRiskPyFuncModel : mlflow.pyfunc.PythonModel wrapper for the pipeline
  - log_pipeline       : log a pipeline to an MLflow experiment
  - load_pipeline      : load a logged pipeline from the Model Registry
  - set_model_alias    : assign a Unity Catalog alias (e.g. 'Champion')

Requires mlflow to be installed:
  pip install medrisk-features[mlflow]
  # or: pip install mlflow

Usage
-----
>>> from medrisk_features.mlflow import log_pipeline, load_pipeline
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

from medrisk_features.mlflow.pyfunc_model import MedRiskPyFuncModel
from medrisk_features.mlflow.tracker import (
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
