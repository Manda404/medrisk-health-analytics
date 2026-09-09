"""Simple public API for end-to-end MedRisk workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from medrisk_health_analytics.pipeline import FeatureEngineeringPipeline
from medrisk_health_analytics.preprocessing import clean_categorical_variables


@dataclass(frozen=True)
class DatasetAnalysis:
    """Compact quality profile returned by :func:`analyze_dataset`."""

    rows: int
    columns: int
    duplicate_rows: int
    dtypes: Mapping[str, str]
    missing_values: Mapping[str, int]
    target_distribution: Mapping[Any, int] | None
    numeric_summary: pd.DataFrame


class DatasetLoader:
    """Load local files, Pandas data, or Unity Catalog tables."""

    def __init__(
        self,
        *,
        spark: Any = None,
        file_format: str | None = None,
        **options: Any,
    ) -> None:
        self.spark = spark
        self.file_format = file_format
        self.options = options

    def load(self, source: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Load one dataset using the configured source options."""
        return load_dataset(
            source,
            spark=self.spark,
            file_format=self.file_format,
            **self.options,
        )


class DatasetAnalyzer:
    """Produce a reusable data-quality profile."""

    def __init__(self, *, target_column: str | None = None) -> None:
        self.target_column = target_column

    def analyze(self, df: pd.DataFrame) -> DatasetAnalysis:
        """Analyze one dataset."""
        return analyze_dataset(df, target_column=self.target_column)


class DatasetPreprocessor:
    """Apply deterministic cleaning before feature engineering."""

    def __init__(self, *, drop_duplicates: bool = True) -> None:
        self.drop_duplicates = drop_duplicates

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the dataset."""
        return preprocess_dataset(df, drop_duplicates=self.drop_duplicates)


class ModelTrainer:
    """Configure and execute boosting training with MLflow."""

    def __init__(self, config: Any = None, **config_options: Any) -> None:
        self.config = config
        self.config_options = config_options
        self.result_: Any = None

    def fit(self, df: pd.DataFrame) -> Any:
        """Train a model and return its MLflow training result."""
        self.result_ = train_model(df, config=self.config, **self.config_options)
        return self.result_


class ModelPredictor:
    """Run inference with one registered MLflow model."""

    def __init__(self, model_uri: str) -> None:
        if not model_uri:
            raise ValueError("model_uri must not be empty")
        self.model_uri = model_uri

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict on one dataset."""
        return predict(self.model_uri, df)


def load_dataset(
    source: str | Path | pd.DataFrame,
    *,
    spark: Any = None,
    file_format: str | None = None,
    **options: Any,
) -> pd.DataFrame:
    """Load a DataFrame, local file, or Unity Catalog table into Pandas."""
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)

    source_name = str(source)
    detected_format = (file_format or Path(source_name).suffix.lstrip(".")).lower()
    if detected_format == "csv":
        return pd.read_csv(source_name, **options)
    if detected_format in {"parquet", "pq"}:
        return pd.read_parquet(source_name, **options)
    if detected_format in {"json", "jsonl"}:
        if detected_format == "jsonl":
            options.setdefault("lines", True)
        return pd.read_json(source_name, **options)
    if spark is not None:
        return spark.table(source_name).toPandas()

    raise ValueError(
        "Unsupported dataset source. Use a CSV, Parquet, JSON file, a Pandas "
        "DataFrame, or provide spark for a Unity Catalog table."
    )


def analyze_dataset(df: pd.DataFrame, *, target_column: str | None = None) -> DatasetAnalysis:
    """Return dataset dimensions, types, missing values, duplicates, and statistics."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target_column is not None and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing")

    target_distribution = None
    if target_column is not None:
        target_distribution = df[target_column].value_counts(dropna=False).to_dict()

    return DatasetAnalysis(
        rows=len(df),
        columns=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        dtypes={str(column): str(dtype) for column, dtype in df.dtypes.items()},
        missing_values={str(column): int(count) for column, count in df.isna().sum().items()},
        target_distribution=target_distribution,
        numeric_summary=df.describe(include=[np.number]).transpose(),
    )


def preprocess_dataset(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Normalize column names, infinities, categories, and duplicate rows."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    result = df.copy(deep=True)
    result.columns = [str(column).strip() for column in result.columns]
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"Column normalization created duplicates: {duplicates}")

    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].replace([np.inf, -np.inf], np.nan)
    if drop_duplicates:
        result = result.drop_duplicates().reset_index(drop=True)
    return clean_categorical_variables(result)


def build_features(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    id_columns: Sequence[str] = (),
    pipeline: FeatureEngineeringPipeline | None = None,
) -> pd.DataFrame:
    """Create MedRisk features while preserving target and identifier columns."""
    transformer = pipeline or FeatureEngineeringPipeline()
    return transformer.transform(df, target_column=target_column, id_columns=id_columns)


def train_model(df: pd.DataFrame, config: Any = None, **config_options: Any) -> Any:
    """Train and log a boosting model with MLflow."""
    from medrisk_health_analytics.boosting import TrainingConfig, train_boosting_model

    resolved_config = config or TrainingConfig(**config_options)
    return train_boosting_model(df=df, config=resolved_config)


def predict(model_uri: str, df: pd.DataFrame) -> pd.DataFrame:
    """Load a registered MLflow model and predict on a Pandas DataFrame."""
    from medrisk_health_analytics.boosting import predict_with_registered_model

    return predict_with_registered_model(model_uri=model_uri, input_df=df)


__all__ = [
    "DatasetAnalysis",
    "DatasetAnalyzer",
    "DatasetLoader",
    "DatasetPreprocessor",
    "ModelPredictor",
    "ModelTrainer",
    "analyze_dataset",
    "build_features",
    "load_dataset",
    "predict",
    "preprocess_dataset",
    "train_model",
]
