# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Prepare Data
# MAGIC
# MAGIC Load the source CSV, clean and enrich patient data, then create the
# MAGIC reproducible train and holdout tables.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Verify runtime

# COMMAND ----------

import platform

import medrisk_health_analytics

print(f"Package version: {medrisk_health_analytics.__version__}")
print(f"Python version: {platform.python_version()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from medrisk_health_analytics import (
    DataQualityValidator,
    DatasetPreprocessor,
    FeatureEngineeringPipeline,
)
from medrisk_health_analytics.boosting import create_train_test_tables
from medrisk_health_analytics.settings import settings

# COMMAND ----------
# MAGIC %md
# MAGIC ## Ingest raw data

# COMMAND ----------

df_raw = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(settings.source_path)
)

(
    df_raw.write.format("delta")
    .mode(settings.write_mode)
    .option("overwriteSchema", "true")
    .saveAsTable(settings.source_table)
)

print(f"Raw table: {settings.source_table} ({df_raw.count():,} rows)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Clean and engineer features

# COMMAND ----------

patient_data = df_raw.toPandas()
patient_data = DatasetPreprocessor().transform(patient_data)

quality_report = DataQualityValidator(
    required_columns=("age", "glucose_fasting", "bmi", settings.target_column),
    target_column=settings.target_column,
    id_columns=settings.id_columns,
    max_missing_ratio=settings.max_missing_ratio,
).validate(patient_data)
quality_report.raise_for_failure()

quality_rows = [
    (settings.environment, "rows", float(quality_report.rows), quality_report.passed),
    (settings.environment, "columns", float(quality_report.columns), quality_report.passed),
    (
        settings.environment,
        "duplicate_rows",
        float(quality_report.duplicate_rows),
        quality_report.passed,
    ),
]
(
    spark.createDataFrame(quality_rows, ["environment", "metric", "value", "passed"])
    .write.format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(settings.data_quality_table)
)
print(f"Data contract passed; metrics appended to {settings.data_quality_table}")

feature_pipeline = FeatureEngineeringPipeline(
    age_group_strategy=settings.age_group_strategy,
    validate_schema=settings.validate_schema,
)
feature_data = feature_pipeline.transform(
    patient_data,
    target_column=settings.target_column,
    id_columns=settings.id_columns,
)

(
    spark.createDataFrame(feature_data)
    .write.format("delta")
    .mode(settings.write_mode)
    .option("overwriteSchema", "true")
    .saveAsTable(settings.feature_table)
)

print(f"Feature table: {settings.feature_table} ({len(feature_data):,} rows)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Create train and holdout tables

# COMMAND ----------

split = create_train_test_tables(
    source_table=settings.feature_table,
    train_table=settings.train_table,
    test_table=settings.test_table,
    target_column=settings.target_column,
    test_size=settings.test_size,
    random_state=settings.random_state,
    stratify=settings.stratify,
    mode=settings.write_mode,
    spark=spark,
)

print(f"Train table: {split.train_table} ({split.train_count:,} rows)")
print(f"Holdout table: {split.test_table} ({split.test_count:,} rows)")

# Build an unlabeled batch so the deployed workflow can verify inference end to end.
(
    spark.table(settings.test_table)
    .drop(settings.target_column)
    .write.format("delta")
    .mode(settings.write_mode)
    .option("overwriteSchema", "true")
    .saveAsTable(settings.scoring_table)
)

print(f"Scoring table: {settings.scoring_table}")
display(spark.table(settings.train_table).limit(10))
