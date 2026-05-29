# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Train And Register Boosting Model
# MAGIC
# MAGIC Objective: train one boosting model from an engineered train table, log
# MAGIC artifacts and metrics to MLflow, and optionally register the model in
# MAGIC Unity Catalog.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("train_table", "workspace.my_schema.patient_train_features")
dbutils.widgets.text("target_column", "diagnosed_diabetes")
dbutils.widgets.text("id_columns", "")
dbutils.widgets.dropdown("model_type", "xgboost", ["xgboost", "catboost", "lightgbm"])
dbutils.widgets.text("experiment_name", "/Shared/experiments/my_schema/boosting")
dbutils.widgets.text("registered_model_name", "workspace.my_schema.boosting_risk_model")
dbutils.widgets.text("run_name", "medrisk-boosting-v1")
dbutils.widgets.text("validation_size", "0.2")
dbutils.widgets.dropdown("use_internal_test_split", "false", ["true", "false"])
dbutils.widgets.text("random_state", "42")

TRAIN_TABLE = dbutils.widgets.get("train_table")
TARGET_COLUMN = dbutils.widgets.get("target_column")
ID_COLUMNS = [c.strip() for c in dbutils.widgets.get("id_columns").split(",") if c.strip()]
MODEL_TYPE = dbutils.widgets.get("model_type")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name") or None
RUN_NAME = dbutils.widgets.get("run_name")
VALIDATION_SIZE = float(dbutils.widgets.get("validation_size"))
USE_INTERNAL_TEST_SPLIT = dbutils.widgets.get("use_internal_test_split").lower() == "true"
RANDOM_STATE = int(dbutils.widgets.get("random_state"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

from medrisk_health_analytics.boosting import TrainingConfig, train_boosting_model

df_train = spark.table(TRAIN_TABLE).toPandas()

config = TrainingConfig(
    target_column=TARGET_COLUMN,
    id_columns=ID_COLUMNS,
    model_type=MODEL_TYPE,
    test_size=VALIDATION_SIZE,
    validation_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    experiment_name=EXPERIMENT_NAME,
    registered_model_name=REGISTERED_MODEL_NAME,
    run_name=RUN_NAME,
    priority_high_threshold=0.8,
    priority_medium_threshold=0.5,
    use_internal_test_split=USE_INTERNAL_TEST_SPLIT,
)

result = train_boosting_model(df=df_train, config=config)

print("Training completed")
print(f"run_id             : {result.run_id}")
print(f"model_uri          : {result.model_uri}")
print(f"registered_model   : {result.registered_model_name}")
print(f"registered_version : {result.registered_model_version}")

dbutils.jobs.taskValues.set(key="model_uri", value=result.model_uri)
dbutils.jobs.taskValues.set(key="registered_model_version", value=result.registered_model_version or "")

for metric in ["test_roc_auc", "test_f1", "test_avg_precision", "test_accuracy"]:
    if metric in result.metrics:
        print(f"{metric:<24}: {result.metrics[metric]:.6f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Metrics

# COMMAND ----------

import pandas as pd

metrics_df = (
    pd.DataFrame(result.metrics.items(), columns=["metric", "value"])
    .assign(split=lambda d: d["metric"].str.split("_").str[0])
    .assign(name=lambda d: d["metric"].str.split("_", n=1).str[1])
    .pivot(index="name", columns="split", values="value")
)

display(metrics_df)
