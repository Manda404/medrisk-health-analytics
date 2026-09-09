"""
train_boosting_model() — high-level training entry point.

This function orchestrates the complete training pipeline:
  1. Auto-detect or use explicit feature/column lists
  2. Split data (random / stratified / temporal)
  3. Fit the TabularPreprocessor
  4. Train the boosting model (XGBoost / CatBoost / LightGBM)
  5. Evaluate on train / valid / test
  6. Log everything to MLflow (params, metrics, artifacts, pyfunc model)
  7. Register in the Model Registry (optional)
  8. Return a TrainingResult

Usage (Databricks notebook)
----------------------------
>>> from medrisk_health_analytics.boosting import train_boosting_model, TrainingConfig
>>>
>>> config = TrainingConfig(
...     target_column="TARGET",
...     id_columns=["client_id", "pers_id"],
...     model_type="xgboost",
...     experiment_name="/Shared/experiments/boosting",
...     registered_model_name="workspace.schema.boosting_model",
... )
>>> result = train_boosting_model(df, config=config)
>>> print(result.metrics)
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

import pandas as pd

from medrisk_health_analytics.boosting.config.schemas import (
    SplitStrategy,
    TaskType,
    TrainingConfig,
    TrainingResult,
)
from medrisk_health_analytics.boosting.models.factory import BoostingModelFactory
from medrisk_health_analytics.boosting.preprocessing.tabular_preprocessor import (
    TabularPreprocessor,
    auto_detect_column_types,
)
from medrisk_health_analytics.boosting.training.evaluator import evaluate_splits


def train_boosting_model(
    df: pd.DataFrame,
    config: Optional[TrainingConfig] = None,
    **config_kwargs,
) -> TrainingResult:
    """
    Train a boosting model end-to-end with full MLflow logging.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataset (all splits combined — function handles
        the split internally). Can be a Pandas DataFrame converted from
        Spark via .toPandas() before calling this function.
    config : TrainingConfig or None
        Full training configuration. If None, a default config is created
        and config_kwargs are forwarded to TrainingConfig(**config_kwargs).
    **config_kwargs
        Keyword arguments forwarded to TrainingConfig if config is None.
        Useful for quick one-liner calls from notebooks.

    Returns
    -------
    TrainingResult
        Dataclass with run_id, model_uri, metrics, feature_names, etc.

    Example
    -------
    >>> result = train_boosting_model(
    ...     df=spark_df.toPandas(),
    ...     target_column="TARGET",
    ...     id_columns=["client_id", "pers_id"],
    ...     model_type="xgboost",
    ...     experiment_name="/Shared/experiments/boosting",
    ... )
    """
    import importlib.util

    if importlib.util.find_spec("mlflow") is None:
        raise ImportError(
            "mlflow is required for training. "
            "Install with: pip install medrisk-health-analytics[mlflow]"
        )

    # ── Resolve configuration ─────────────────────────────────────────────────
    if config is None:
        config = TrainingConfig(**config_kwargs)

    required_columns = {config.target_column, *config.id_columns}
    if config.temporal_column:
        required_columns.add(config.temporal_column)
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")
    if df.columns.duplicated().any():
        duplicates = sorted(
            {
                str(column)
                for column, duplicated in zip(df.columns, df.columns.duplicated(), strict=True)
                if duplicated
            }
        )
        raise ValueError(f"Training data contains duplicate columns: {duplicates}")
    if df.empty:
        raise ValueError("Training data must contain at least one row")
    if config.split_strategy == SplitStrategy.TEMPORAL:
        temporal_column = config.temporal_column
        assert temporal_column is not None
        df = df.sort_values(temporal_column, kind="stable").reset_index(drop=True)

    excluded_columns = list(config.exclude_columns)
    if config.temporal_column and config.temporal_column not in excluded_columns:
        excluded_columns.append(config.temporal_column)

    # ── Auto-detect column types if not explicitly provided ──────────────────
    numeric_cols, categorical_cols = auto_detect_column_types(
        df,
        exclude=excluded_columns,
        target=config.target_column,
        id_columns=config.id_columns,
    )

    if config.feature_columns is not None:
        missing_features = sorted(set(config.feature_columns) - set(df.columns))
        if missing_features:
            raise ValueError(f"Training data is missing configured features: {missing_features}")
        selected = df[config.feature_columns]
        numeric_cols = selected.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = selected.select_dtypes(exclude=["number"]).columns.tolist()

    if config.numeric_columns:
        numeric_cols = config.numeric_columns
    if config.categorical_columns:
        categorical_cols = config.categorical_columns

    raw_feature_cols = numeric_cols + categorical_cols
    if not raw_feature_cols:
        raise ValueError("Training data does not contain any usable feature columns")

    # ── Build feature matrix and target ──────────────────────────────────────
    X = df[raw_feature_cols].copy()
    y = df[config.target_column].copy()
    if y.isna().any():
        raise ValueError(f"Target column '{config.target_column}' contains missing values")
    class_count = y.nunique(dropna=True)
    if class_count < 2:
        raise ValueError(
            f"Target column '{config.target_column}' must contain at least two classes"
        )
    if config.task_type == TaskType.BINARY_CLASSIFICATION and class_count != 2:
        raise ValueError(
            f"Binary classification requires exactly two target classes; found {class_count}"
        )

    # ── Split dataset ─────────────────────────────────────────────────────────
    # In Databricks/Unity Catalog workflows, the real holdout test set should
    # usually be materialized as its own table before this trainer runs.
    if config.use_internal_test_split:
        X_train, X_test, y_train, y_test = _split_data(X, y, config)
    else:
        X_train, y_train = X, y
        X_test = y_test = None

    X_train, X_valid, y_train, y_valid = _split_data(
        X_train,
        y_train,
        config,
        test_size=config.validation_size,
    )

    # ── Fit preprocessor ──────────────────────────────────────────────────────
    preprocessor = TabularPreprocessor(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        numeric_impute_strategy=config.numeric_impute_strategy,
        categorical_impute_strategy=config.categorical_impute_strategy,
        categorical_encoding=config.categorical_encoding,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)
    transformed_feature_cols = preprocessor.feature_names_out

    # Wrap back as DataFrame (preserves feature names for tree models)
    X_train_df = pd.DataFrame(X_train_t, columns=transformed_feature_cols, index=X_train.index)
    X_valid_df = pd.DataFrame(X_valid_t, columns=transformed_feature_cols, index=X_valid.index)

    # ── Train the boosting model ──────────────────────────────────────────────
    model = BoostingModelFactory.create(
        model_type=config.model_type,
        params=config.model_params,
        task_type=config.task_type.value,
    )
    model.fit(X_train_df, y_train, X_valid=X_valid_df, y_valid=y_valid)

    # ── Evaluate on all splits ────────────────────────────────────────────────
    splits = {
        "train": {"X": X_train_df, "y": y_train},
        "valid": {"X": X_valid_df, "y": y_valid},
    }
    if X_test is not None and y_test is not None:
        X_test_t = preprocessor.transform(X_test)
        X_test_df = pd.DataFrame(X_test_t, columns=transformed_feature_cols, index=X_test.index)
        splits["test"] = {"X": X_test_df, "y": y_test}

    metrics = evaluate_splits(model, splits, task_type=config.task_type.value)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    result = _log_to_mlflow(
        model=model,
        preprocessor=preprocessor,
        config=config,
        raw_feature_names=raw_feature_cols,
        transformed_feature_names=transformed_feature_cols,
        metrics=metrics,
        df_sample=X_train.head(10),
        training_dataset=pd.concat(
            [X_train, y_train.rename(config.target_column)],
            axis=1,
        ),
    )
    result.config = config

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
    test_size: Optional[float] = None,
) -> tuple:
    """Split X/y according to config.split_strategy."""
    # Lazy import — sklearn is an optional dependency
    from sklearn.model_selection import train_test_split

    size = test_size if test_size is not None else config.test_size
    strategy = config.split_strategy

    if strategy == SplitStrategy.STRATIFIED:
        return train_test_split(
            X,
            y,
            test_size=size,
            random_state=config.random_state,
            stratify=y,
        )
    elif strategy == SplitStrategy.TEMPORAL:
        # Temporal split: last `size` fraction is test
        split_idx = int(len(X) * (1 - size))
        return (
            X.iloc[:split_idx],
            X.iloc[split_idx:],
            y.iloc[:split_idx],
            y.iloc[split_idx:],
        )
    else:
        return train_test_split(
            X,
            y,
            test_size=size,
            random_state=config.random_state,
        )


def _log_to_mlflow(
    model,
    preprocessor: TabularPreprocessor,
    config: TrainingConfig,
    raw_feature_names: list,
    transformed_feature_names: list,
    metrics: dict,
    df_sample: Optional[pd.DataFrame] = None,
    training_dataset: Optional[pd.DataFrame] = None,
) -> TrainingResult:
    """
    Serialize artifacts, open an MLflow run, and log everything.

    Logged artifacts
    ----------------
    - model.{json|cbm|txt}    : native boosting model
    - preprocessor.pkl        : fitted TabularPreprocessor
    - config.json             : full training configuration
    - feature_names.json      : ordered raw input feature column list
    - transformed_feature_names.json : ordered model feature column list
    - feature_importance.csv  : feature importance ranking
    - The pyfunc model (mlflow.pyfunc.log_model)
    """
    import mlflow
    import mlflow.pyfunc

    from medrisk_health_analytics import __version__
    from medrisk_health_analytics.boosting.pyfunc.boosting_pyfunc_model import BoostingPyFuncModel
    from medrisk_health_analytics.mlflow.signature import make_nullable_safe_sample

    mlflow.set_experiment(config.experiment_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Serialize model ───────────────────────────────────────────────
        model_ext = {"xgboost": ".json", "catboost": ".cbm", "lightgbm": ".txt"}.get(
            config.model_type.lower(), ".pkl"
        )
        model_path = os.path.join(tmpdir, f"model{model_ext}")
        model.save(os.path.splitext(model_path)[0])  # Bug #7 fix: safe extension removal

        # ── Serialize preprocessor ────────────────────────────────────────
        preprocessor_path = os.path.join(tmpdir, "preprocessor.pkl")
        preprocessor.save(preprocessor_path)

        # ── Config JSON ───────────────────────────────────────────────────
        config_dict = config.to_mlflow_params()
        config_dict["medrisk_version"] = __version__
        config_dict["model_type"] = config.model_type
        config_dict["id_columns_list"] = config.id_columns
        config_dict["model_version"] = "unregistered"
        config_dict["registered_model_name"] = config.registered_model_name

        # Priority thresholds needed at inference
        config_dict["priority_high_threshold"] = config.priority_high_threshold
        config_dict["priority_medium_threshold"] = config.priority_medium_threshold
        config_dict["id_columns"] = config.id_columns

        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)

        # ── Feature names JSON ────────────────────────────────────────────
        fn_path = os.path.join(tmpdir, "feature_names.json")
        with open(fn_path, "w", encoding="utf-8") as f:
            json.dump(raw_feature_names, f, indent=2)

        transformed_fn_path = os.path.join(tmpdir, "transformed_feature_names.json")
        with open(transformed_fn_path, "w", encoding="utf-8") as f:
            json.dump(transformed_feature_names, f, indent=2)

        # ── Feature importance CSV ────────────────────────────────────────
        fi_path: Optional[str] = os.path.join(tmpdir, "feature_importance.csv")
        try:
            fi = model.get_feature_importance()
            fi.to_csv(fi_path)
        except Exception:
            fi_path = None

        # ── Start MLflow run ──────────────────────────────────────────────
        run_tags = {
            "model_type": config.model_type,
            "task_type": config.task_type.value,
            "medrisk_version": __version__,
            "deployment_environment": config.environment or "unknown",
            "training_data_source": config.training_data_source or "unspecified",
        }

        with mlflow.start_run(run_name=config.run_name, tags=run_tags) as run:
            run_id = run.info.run_id

            # Log parameters
            safe_params = {
                k: (str(v)[:250] if isinstance(v, list | dict) else v)
                for k, v in config.to_mlflow_params().items()
            }
            mlflow.log_params(safe_params)

            if config.training_data_source:
                mlflow.set_tag("mlflow.data.context", "training")
            if training_dataset is not None:
                dataset = mlflow.data.from_pandas(
                    training_dataset,
                    name="medrisk_training_features",
                    targets=config.target_column,
                )
                mlflow.log_input(dataset, context="training")

            # Keep the MLflow overview focused; detailed metrics remain in the result.
            mlflow.log_metrics(_select_mlflow_metrics(metrics))

            # Log raw artifacts
            mlflow.log_artifact(config_path, artifact_path="pipeline_metadata")
            mlflow.log_artifact(fn_path, artifact_path="pipeline_metadata")
            mlflow.log_artifact(transformed_fn_path, artifact_path="pipeline_metadata")
            if fi_path and os.path.exists(fi_path):
                mlflow.log_artifact(fi_path, artifact_path="pipeline_metadata")

            # Infer MLflow model signature
            signature = None
            input_example = None
            if df_sample is not None:
                from mlflow.models.signature import infer_signature

                signature_input = make_nullable_safe_sample(df_sample)
                dummy_proba = model.predict_proba(
                    pd.DataFrame(
                        preprocessor.transform(signature_input),
                        columns=transformed_feature_names,
                    )
                )
                signature = infer_signature(signature_input, dummy_proba)
                if config.log_input_example:
                    input_example = signature_input

            # Log the full pyfunc model
            pyfunc_instance = BoostingPyFuncModel()

            model_requirement = {
                "xgboost": "xgboost",
                "catboost": "catboost",
                "lightgbm": "lightgbm",
            }.get(config.model_type, "scikit-learn>=1.3")

            mlflow.pyfunc.log_model(
                name=config.artifact_path,
                python_model=pyfunc_instance,
                artifacts={
                    "model": model_path,
                    "preprocessor": preprocessor_path,
                    "config": config_path,
                    "feature_names": fn_path,
                    "transformed_feature_names": transformed_fn_path,
                },
                signature=signature,
                input_example=input_example,
                pip_requirements=[
                    f"medrisk-health-analytics=={__version__}",
                    model_requirement,
                    "pandas>=2.0",
                    "numpy>=1.24",
                    "scikit-learn>=1.3",
                ],
            )

            model_uri = f"runs:/{run_id}/{config.artifact_path}"
            artifact_base_uri = f"runs:/{run_id}/pipeline_metadata"

            # Register in Model Registry (optional)
            registered_version = None
            if config.registered_model_name:
                reg = mlflow.register_model(
                    model_uri=model_uri,
                    name=config.registered_model_name,
                )
                registered_version = reg.version

    return TrainingResult(
        run_id=run_id,
        model_uri=model_uri,
        registered_model_name=config.registered_model_name,
        registered_model_version=registered_version,
        metrics=metrics,
        feature_names=raw_feature_names,
        artifact_paths={
            "pyfunc_model": model_uri,
            "pipeline_metadata": artifact_base_uri,
            "config": f"{artifact_base_uri}/config.json",
            "feature_names": f"{artifact_base_uri}/feature_names.json",
            "transformed_feature_names": f"{artifact_base_uri}/transformed_feature_names.json",
        },
    )


def _select_mlflow_metrics(metrics: dict) -> dict:
    """Return the small set of metrics used to compare training runs."""
    tracked_names = (
        "valid_roc_auc",
        "valid_avg_precision",
        "valid_recall",
        "valid_specificity",
        "valid_mcc",
        "valid_brier_score",
    )
    return {name: metrics[name] for name in tracked_names if name in metrics}
