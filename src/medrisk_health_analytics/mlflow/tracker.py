"""
MLflow tracking utilities for medrisk-health-analytics pipelines.

Provides high-level functions to:
  - serialize and log a FeatureEngineeringPipeline as a pyfunc model
  - register the logged model in the MLflow Model Registry
  - load a registered pipeline back for inference

These utilities are designed for use inside Databricks notebooks or
any environment where MLflow is configured (local, remote, Unity Catalog).

Typical workflow
----------------
1. Build and run the pipeline:
   >>> pipeline = FeatureEngineeringPipeline(age_group_strategy="detailed")
   >>> df_enriched = pipeline.transform(df_train)

2. Log to MLflow:
   >>> result = log_pipeline(
   ...     pipeline=pipeline,
   ...     df_sample=df_train.head(5),
   ...     experiment_name="/Shared/experiments/medrisk",
   ...     registered_model_name="workspace.schema.medrisk_pipeline",
   ... )
   >>> print(result.model_uri)

3. Load for inference:
   >>> loaded = load_pipeline(result.model_uri)
   >>> df_out = loaded.predict(df_new)
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# mlflow is an optional dependency — imported lazily in each function
_MLFLOW_NOT_INSTALLED_MSG = (
    "mlflow is required for this feature. Install it with:\n"
    "  pip install medrisk-health-analytics[mlflow]\n"
    "  # or: pip install mlflow"
)


@dataclass
class PipelineLogResult:
    """
    Result returned by log_pipeline().

    Attributes
    ----------
    run_id : str
        MLflow run ID of the experiment run.
    model_uri : str
        URI to the logged model (e.g. 'runs:/<run_id>/medrisk_pipeline').
    registered_model_name : str or None
        Name of the registered model in the Model Registry, if any.
    registered_model_version : str or None
        Version number of the registered model, if any.
    feature_names : list[str]
        Output feature column names produced by the pipeline.
    config : dict
        Pipeline configuration that was logged.
    artifact_paths : dict[str, str]
        Mapping of artifact keys to their local file paths.
    """

    run_id: str
    model_uri: str
    registered_model_name: Optional[str] = None
    registered_model_version: Optional[str] = None
    feature_names: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    artifact_paths: dict = field(default_factory=dict)


def log_pipeline(
    pipeline,
    experiment_name: str,
    artifact_path: str = "medrisk_pipeline",
    df_sample: Optional[pd.DataFrame] = None,
    registered_model_name: Optional[str] = None,
    tags: Optional[dict] = None,
    run_name: Optional[str] = None,
) -> PipelineLogResult:
    """
    Log a FeatureEngineeringPipeline to MLflow as a pyfunc model.

    The function serializes the pipeline, its configuration and the list
    of output features, then logs everything inside a single MLflow run.
    Optionally registers the model in the Model Registry.

    Parameters
    ----------
    pipeline : FeatureEngineeringPipeline
        Fitted (or configured) pipeline instance to log.
    experiment_name : str
        MLflow experiment name or path (e.g. '/Shared/experiments/medrisk').
    artifact_path : str, default 'medrisk_pipeline'
        Sub-path within the run's artifact store where the model is saved.
    df_sample : pd.DataFrame or None
        Optional small sample DataFrame used to infer the MLflow signature
        and log input/output examples. Recommended for production.
    registered_model_name : str or None
        If provided, registers the model under this name in the Model Registry.
        For Unity Catalog use the three-level name: 'catalog.schema.model_name'.
    tags : dict or None
        Additional key/value tags to attach to the MLflow run.
    run_name : str or None
        Optional display name for the MLflow run.

    Returns
    -------
    PipelineLogResult
        Dataclass containing run_id, model_uri, registered model info,
        feature names and artifact paths.

    Raises
    ------
    ImportError
        If mlflow is not installed.

    Example
    -------
    >>> result = log_pipeline(
    ...     pipeline=pipeline,
    ...     experiment_name="/Shared/experiments/medrisk",
    ...     df_sample=df.head(10),
    ...     registered_model_name="workspace.analytics.medrisk_pipeline",
    ... )
    >>> print(result.model_uri)
    'runs:/abc123/medrisk_pipeline'
    """
    try:
        import mlflow
        import mlflow.pyfunc
    except ImportError as exc:
        raise ImportError(_MLFLOW_NOT_INSTALLED_MSG) from exc

    from medrisk_health_analytics.mlflow.pyfunc_model import MedRiskPyFuncModel
    from medrisk_health_analytics.mlflow.signature import make_nullable_safe_sample

    # -- Build pipeline configuration dict --------------------------------
    config = {
        "age_group_strategy": getattr(pipeline, "age_group_strategy", "detailed"),
        "validate_schema": getattr(pipeline, "validate_schema", True),
        "medrisk_version": _get_package_version(),
    }

    # -- Compute output feature names using df_sample ---------------------
    feature_names: list = []
    if df_sample is not None:
        try:
            df_out = pipeline.transform(df_sample.copy())
            feature_names = df_out.columns.tolist()
        except Exception:
            feature_names = []

    # -- Serialize artifacts to a temp directory --------------------------
    artifact_paths: dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pipeline pickle
        pipeline_path = os.path.join(tmpdir, "pipeline.pkl")
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)
        artifact_paths["pipeline"] = pipeline_path

        # Config JSON
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        artifact_paths["config"] = config_path

        # Feature names JSON
        feature_names_path = os.path.join(tmpdir, "feature_names.json")
        with open(feature_names_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)
        artifact_paths["feature_names"] = feature_names_path

        # -- Set MLflow experiment -----------------------------------------
        mlflow.set_experiment(experiment_name)

        # -- Infer MLflow signature (optional) ----------------------------
        signature = None
        input_example = None
        if df_sample is not None and len(feature_names) > 0:
            try:
                from mlflow.models.signature import infer_signature

                signature_input = make_nullable_safe_sample(df_sample)
                df_out_sample = make_nullable_safe_sample(
                    pipeline.transform(signature_input.copy())
                )
                signature = infer_signature(signature_input, df_out_sample)
                input_example = signature_input.head(3)
            except Exception:
                pass  # Signature inference is best-effort

        # -- Start MLflow run and log everything --------------------------
        run_tags = {"medrisk_version": _get_package_version()}
        if tags:
            run_tags.update(tags)

        with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
            run_id = run.info.run_id

            # Log configuration parameters
            mlflow.log_params(config)

            # Log feature count as a metric
            if feature_names:
                mlflow.log_metric("n_output_features", len(feature_names))

            # Log raw artifacts (config + feature names)
            mlflow.log_artifact(config_path, artifact_path="pipeline_metadata")
            mlflow.log_artifact(feature_names_path, artifact_path="pipeline_metadata")

            # Get the correct pyfunc class (with mlflow.pyfunc.PythonModel base)
            pyfunc_instance = MedRiskPyFuncModel()

            # Log the model as mlflow.pyfunc
            mlflow.pyfunc.log_model(
                name=artifact_path,
                python_model=pyfunc_instance,
                artifacts={
                    "pipeline": pipeline_path,
                    "config": config_path,
                    "feature_names": feature_names_path,
                },
                signature=signature,
                input_example=input_example,
                pip_requirements=[
                    f"medrisk-health-analytics=={_get_package_version()}",
                    "pandas>=2.0",
                    "numpy>=1.24",
                ],
            )

            model_uri = f"runs:/{run_id}/{artifact_path}"
            metadata_uri = f"runs:/{run_id}/pipeline_metadata"

            # -- Optionally register in Model Registry --------------------
            registered_version = None
            if registered_model_name:
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=registered_model_name,
                )
                registered_version = registered_model.version

            # Bug #6 fix: store MLflow URIs (not deleted temp file paths)
            mlflow_artifact_paths = {
                "pyfunc_model": model_uri,
                "pipeline_metadata": metadata_uri,
                "config": f"{metadata_uri}/config.json",
                "feature_names": f"{metadata_uri}/feature_names.json",
            }

    return PipelineLogResult(
        run_id=run_id,
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        registered_model_version=registered_version,
        feature_names=feature_names,
        config=config,
        artifact_paths=mlflow_artifact_paths,
    )


def load_pipeline(model_uri: str):
    """
    Load a logged FeatureEngineeringPipeline from MLflow.

    Accepts any valid MLflow model URI:
    - 'runs:/<run_id>/medrisk_pipeline'
    - 'models:/workspace.schema.medrisk_pipeline/Production'
    - 'models:/workspace.schema.medrisk_pipeline@Champion'

    Parameters
    ----------
    model_uri : str
        MLflow model URI pointing to a logged MedRiskPyFuncModel.

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        Loaded model. Call .predict(df) to apply the pipeline.

    Raises
    ------
    ImportError
        If mlflow is not installed.

    Example
    -------
    >>> loaded = load_pipeline("models:/workspace.schema.medrisk_pipeline@Champion")
    >>> df_enriched = loaded.predict(df_raw)
    """
    try:
        import mlflow.pyfunc
    except ImportError as exc:
        raise ImportError(_MLFLOW_NOT_INSTALLED_MSG) from exc

    return mlflow.pyfunc.load_model(model_uri)


def set_model_alias(
    registered_model_name: str,
    version: str,
    alias: str,
) -> None:
    """
    Assign an alias to a registered model version (Unity Catalog style).

    Parameters
    ----------
    registered_model_name : str
        Three-level model name, e.g. 'workspace.schema.medrisk_pipeline'.
    version : str
        Model version number as a string, e.g. '3'.
    alias : str
        Alias to assign, e.g. 'Champion' or 'Challenger'.

    Example
    -------
    >>> set_model_alias(
    ...     "workspace.analytics.medrisk_pipeline",
    ...     version="3",
    ...     alias="Champion",
    ... )
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(_MLFLOW_NOT_INSTALLED_MSG) from exc

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=version,
    )


def _get_package_version() -> str:
    """Return the installed medrisk-health-analytics version string."""
    try:
        from medrisk_health_analytics import __version__

        return __version__
    except ImportError:
        return "unknown"
