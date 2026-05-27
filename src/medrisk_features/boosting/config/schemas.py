"""
Configuration dataclasses for the medrisk-features boosting MLOps layer.

Design rationale
----------------
Using Python dataclasses (rather than plain dicts) gives:
  - auto-completion in IDEs and Databricks notebooks
  - explicit field documentation
  - default values for optional parameters
  - easy serialization to/from JSON for MLflow logging

All configs are intentionally flat — no nested objects — so they can be
serialized with mlflow.log_params() without custom adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SplitStrategy(str, Enum):
    """Dataset split strategy."""

    RANDOM = "random"  # standard random split
    STRATIFIED = "stratified"  # stratified by target (classification)
    TEMPORAL = "temporal"  # chronological split by date column


class TaskType(str, Enum):
    """ML task type."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """
    Full configuration for a boosting model training run.

    Designed to be passed directly to train_boosting_model() and logged
    as MLflow parameters at the start of a training run.

    Example (Databricks notebook)
    ------------------------------
    >>> config = TrainingConfig(
    ...     target_column="TARGET",
    ...     id_columns=["client_id", "pers_id"],
    ...     model_type="xgboost",
    ...     experiment_name="/Shared/experiments/boosting",
    ...     registered_model_name="workspace.schema.boosting_model",
    ... )
    """

    # ── Data ──────────────────────────────────────────────────────────────
    target_column: str = "TARGET"
    """Name of the binary or multiclass target column."""

    id_columns: List[str] = field(default_factory=list)
    """Identifier columns preserved in the output (not used as features)."""

    feature_columns: Optional[List[str]] = None
    """Explicit list of feature columns. If None, all non-id/non-target cols."""

    exclude_columns: List[str] = field(default_factory=list)
    """Columns to drop before training (dates, free text, leaky columns)."""

    numeric_columns: Optional[List[str]] = None
    """Numeric feature columns. Auto-detected if None."""

    categorical_columns: Optional[List[str]] = None
    """Categorical feature columns. Auto-detected if None."""

    # ── Split ─────────────────────────────────────────────────────────────
    split_strategy: SplitStrategy = SplitStrategy.STRATIFIED
    """How to split the dataset: random, stratified, or temporal."""

    test_size: float = 0.2
    """Proportion of data reserved for the test set (0.0–1.0)."""

    validation_size: float = 0.2
    """Proportion of training data used as validation set (0.0–1.0)."""

    use_internal_test_split: bool = True
    """
    Whether train_boosting_model() should create an internal test split.

    Set to False when a separate holdout test table already exists in
    Unity Catalog. In that case the trainer creates only train/validation
    splits, and the holdout table should be evaluated separately.
    """

    random_state: int = 42
    """Random seed for reproducibility."""

    temporal_column: Optional[str] = None
    """Column name used for temporal split (requires split_strategy='temporal')."""

    # ── Preprocessing ─────────────────────────────────────────────────────
    numeric_impute_strategy: str = "median"
    """Imputation strategy for numeric columns: 'mean', 'median', 'constant'."""

    categorical_impute_strategy: str = "most_frequent"
    """Imputation strategy for categorical columns."""

    categorical_encoding: str = "ordinal"
    """Encoding strategy: 'ordinal' (for tree models) or 'onehot'."""

    # ── Model ─────────────────────────────────────────────────────────────
    model_type: str = "xgboost"
    """Boosting model type: 'xgboost', 'catboost', or 'lightgbm'."""

    task_type: TaskType = TaskType.BINARY_CLASSIFICATION
    """ML task: binary_classification or multiclass_classification."""

    model_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
    )
    """Hyperparameters passed directly to the chosen boosting model."""

    # ── MLflow ────────────────────────────────────────────────────────────
    experiment_name: str = "/Shared/experiments/boosting_experiment"
    """Full MLflow experiment path."""

    run_name: Optional[str] = None
    """Optional display name for the MLflow run."""

    artifact_path: str = "boosting_model"
    """Sub-path in the MLflow run artifact store for the pyfunc model."""

    registered_model_name: Optional[str] = None
    """Unity Catalog model name: 'catalog.schema.model_name'. None = no registration."""

    log_input_example: bool = True
    """Whether to log a sample of input data as an MLflow artifact."""

    # ── Inference / output ────────────────────────────────────────────────
    priority_high_threshold: float = 0.8
    """Probability threshold above which priority = 'HIGH'."""

    priority_medium_threshold: float = 0.5
    """Probability threshold above which priority = 'MEDIUM' (else 'LOW')."""

    output_columns: List[str] = field(
        default_factory=lambda: [
            "probability",
            "priority",
            "prediction_date",
            "model_version",
        ]
    )
    """Columns included in the prediction output (id_columns are always added)."""

    def to_mlflow_params(self) -> Dict[str, Any]:
        """
        Flatten configuration to a dict suitable for mlflow.log_params().

        MLflow requires all param values to be strings or numbers, so
        complex types (lists, dicts) are serialized to strings.

        Returns
        -------
        dict
            Flat key/value dict safe to pass to mlflow.log_params().
        """
        import json

        return {
            "target_column": self.target_column,
            "id_columns": json.dumps(self.id_columns),
            "model_type": self.model_type,
            "task_type": self.task_type.value,
            "split_strategy": self.split_strategy.value,
            "test_size": self.test_size,
            "validation_size": self.validation_size,
            "use_internal_test_split": self.use_internal_test_split,
            "random_state": self.random_state,
            "numeric_impute_strategy": self.numeric_impute_strategy,
            "categorical_encoding": self.categorical_encoding,
            "priority_high_threshold": self.priority_high_threshold,
            "priority_medium_threshold": self.priority_medium_threshold,
            **{f"model__{k}": v for k, v in self.model_params.items()},
        }


# ---------------------------------------------------------------------------
# InferenceConfig
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """
    Configuration for a batch inference run.

    Example
    -------
    >>> cfg = InferenceConfig(
    ...     model_uri="models:/workspace.schema.boosting_model@Champion",
    ...     input_table="workspace.schema.scoring_data",
    ...     output_table="workspace.schema.scoring_results",
    ... )
    """

    model_uri: str = ""
    """MLflow model URI. Supports: 'runs:/', 'models:/', Unity Catalog aliases."""

    input_table: Optional[str] = None
    """Unity Catalog table to score: 'catalog.schema.table'. Used if no input_df."""

    output_table: Optional[str] = None
    """Unity Catalog table to write results to. None = return DataFrame only."""

    output_mode: str = "overwrite"
    """Spark write mode: 'overwrite', 'append', 'merge'."""

    id_columns: List[str] = field(default_factory=list)
    """Identifier columns to preserve in output."""

    batch_size: Optional[int] = None
    """Process input in batches of this size. None = load all at once."""


# ---------------------------------------------------------------------------
# TrainingResult
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """
    Result returned by train_boosting_model().

    Contains all information needed to reproduce, evaluate and deploy
    the trained model.
    """

    run_id: str
    """MLflow run ID of the training run."""

    model_uri: str
    """URI of the logged pyfunc model: 'runs:/<run_id>/<artifact_path>'."""

    registered_model_name: Optional[str] = None
    """Name in the Model Registry if the model was registered."""

    registered_model_version: Optional[str] = None
    """Version number in the Model Registry."""

    metrics: Dict[str, float] = field(default_factory=dict)
    """All computed metrics (train/valid/test splits, all metric names)."""

    feature_names: List[str] = field(default_factory=list)
    """Ordered list of feature columns used for training."""

    artifact_paths: Dict[str, str] = field(default_factory=dict)
    """Mapping of artifact keys to their MLflow artifact URIs."""

    config: Optional[TrainingConfig] = None
    """The TrainingConfig used for this run (for full reproducibility)."""
