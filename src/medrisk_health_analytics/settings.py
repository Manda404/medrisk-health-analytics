"""Central, validated runtime configuration for local and Databricks workloads."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, Mapping, Optional, Tuple

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    """Settings loaded from environment variables prefixed with ``MEDRISK_``."""

    model_config = SettingsConfigDict(
        env_prefix="MEDRISK_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    environment: Literal["dev", "stg", "prd"] = "dev"
    catalog: str = "workspace"
    schema_name: str = Field(
        default="mlops_dev", validation_alias=AliasChoices("MEDRISK_SCHEMA", "schema")
    )
    source_path: str = "/Volumes/workspace/mlops_dev/medrisk_data/patient.csv"
    target_column: str = "diagnosed_diabetes"
    id_columns: Tuple[str, ...] = ()
    write_mode: Literal["overwrite", "append"] = "overwrite"
    age_group_strategy: Literal["detailed", "coarse"] = "detailed"
    validate_schema: bool = True
    test_size: float = Field(default=0.2, ge=0, le=1)
    validation_size: float = Field(default=0.2, ge=0, le=1)
    random_state: int = 42
    stratify: bool = True
    model_type: Literal[
        "xgboost",
        "catboost",
        "lightgbm",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ] = "xgboost"
    mlflow_experiment_name: str = "/Shared/medrisk-health-analytics"
    model_name: str = "boosting_risk_model"
    model_alias: str = "Champion"
    run_name: str = "medrisk-boosting"
    use_internal_test_split: bool = False
    write_holdout_predictions: bool = True
    min_roc_auc: float = Field(default=0.70, ge=0, le=1)
    min_recall: float = Field(default=0.60, ge=0, le=1)
    max_missing_ratio: float = Field(default=0.05, ge=0, le=1)
    drift_psi_threshold: float = Field(default=0.20, gt=0)
    fail_on_drift: bool = False
    log_directory: Path = Path("logs")
    log_level: str = "INFO"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"

    _ENV_PREFIX: ClassVar[str] = "MEDRISK_"

    @field_validator("catalog", "schema_name", "model_name", "log_level")
    @classmethod
    def reject_empty_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("id_columns", mode="before")
    @classmethod
    def parse_id_columns(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @property
    def schema(self) -> str:
        """Unity Catalog schema name (compatibility accessor)."""
        return self.schema_name

    @property
    def table_prefix(self) -> str:
        return f"{self.catalog}.{self.schema_name}"

    @property
    def source_table(self) -> str:
        return f"{self.table_prefix}.patient_raw_data"

    @property
    def feature_table(self) -> str:
        return f"{self.table_prefix}.patient_features"

    @property
    def train_table(self) -> str:
        return f"{self.table_prefix}.patient_train_features"

    @property
    def test_table(self) -> str:
        return f"{self.table_prefix}.patient_test_features"

    @property
    def holdout_predictions_table(self) -> str:
        return f"{self.table_prefix}.patient_test_predictions"

    @property
    def scoring_table(self) -> str:
        return f"{self.table_prefix}.patient_scoring_data"

    @property
    def scoring_predictions_table(self) -> str:
        return f"{self.table_prefix}.patient_scoring_predictions"

    @property
    def data_quality_table(self) -> str:
        return f"{self.table_prefix}.data_quality_metrics"

    @property
    def drift_metrics_table(self) -> str:
        return f"{self.table_prefix}.feature_drift_metrics"

    @property
    def registered_model_name(self) -> str:
        return f"{self.table_prefix}.{self.model_name}"

    @property
    def registered_model_uri(self) -> str:
        return f"models:/{self.registered_model_name}@{self.model_alias}"

    @property
    def log_file(self) -> Path:
        return self.log_directory / "Medrisk-Health-Analytics.log"

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> RuntimeSettings:
        """Load the process environment or an explicit mapping (useful in tests)."""
        if environ is None:
            return cls()
        field_names = set(cls.model_fields)
        values = {}
        for key, value in environ.items():
            if not key.upper().startswith(cls._ENV_PREFIX):
                continue
            field_name = key[len(cls._ENV_PREFIX) :].lower()
            if field_name == "schema":
                field_name = "schema_name"
            if field_name in field_names:
                values[field_name] = value
        return cls(**values)


settings = RuntimeSettings()

__all__ = ["RuntimeSettings", "settings"]
