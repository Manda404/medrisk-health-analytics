# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Evaluate Registered Model On Holdout Test Table
# MAGIC
# MAGIC Objective: load a registered model, score an engineered holdout test table,
# MAGIC compute evaluation metrics, and optionally persist predictions.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("model_uri", "models:/workspace.my_schema.boosting_risk_model@Champion")
dbutils.widgets.text("test_table", "workspace.my_schema.patient_test_features")
dbutils.widgets.text("target_column", "diagnosed_diabetes")
dbutils.widgets.text("output_table", "workspace.my_schema.patient_test_predictions")
dbutils.widgets.dropdown("write_predictions", "false", ["true", "false"])
dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "append"])

MODEL_URI = dbutils.widgets.get("model_uri")
TEST_TABLE = dbutils.widgets.get("test_table")
TARGET_COLUMN = dbutils.widgets.get("target_column")
OUTPUT_TABLE = dbutils.widgets.get("output_table")
WRITE_PREDICTIONS = dbutils.widgets.get("write_predictions").lower() == "true"
MODE = dbutils.widgets.get("mode")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Score Holdout Table

# COMMAND ----------

import pandas as pd
from medrisk_health_analytics.boosting import evaluate_registered_model_on_holdout

result = evaluate_registered_model_on_holdout(
    model_uri=MODEL_URI,
    input_table=TEST_TABLE,
    target_column=TARGET_COLUMN,
    output_table=OUTPUT_TABLE if WRITE_PREDICTIONS else None,
    mode=MODE,
    spark=spark,
)

metrics_df = pd.DataFrame(result.metrics.items(), columns=["metric", "value"])
display(metrics_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional - Save Predictions

# COMMAND ----------

if WRITE_PREDICTIONS:
    print(f"Predictions written to {OUTPUT_TABLE}")

display(result.predictions.head(10))
