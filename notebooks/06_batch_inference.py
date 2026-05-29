# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Batch Inference
# MAGIC
# MAGIC Objective: read a scoring table, apply feature engineering, score with a
# MAGIC registered model, and save predictions to Unity Catalog.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("model_uri", "models:/workspace.my_schema.boosting_risk_model@Champion")
dbutils.widgets.text("input_table", "workspace.my_schema.patient_scoring_raw")
dbutils.widgets.text("output_table", "workspace.my_schema.patient_scoring_predictions")
dbutils.widgets.text("id_columns", "")
dbutils.widgets.text("target_column", "diagnosed_diabetes")
dbutils.widgets.dropdown("age_group_strategy", "detailed", ["detailed", "coarse"])
dbutils.widgets.dropdown("validate_schema", "true", ["true", "false"])
dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "append"])

MODEL_URI = dbutils.widgets.get("model_uri")
INPUT_TABLE = dbutils.widgets.get("input_table")
OUTPUT_TABLE = dbutils.widgets.get("output_table")
ID_COLUMNS = [c.strip() for c in dbutils.widgets.get("id_columns").split(",") if c.strip()]
TARGET_COLUMN = dbutils.widgets.get("target_column")
AGE_GROUP_STRATEGY = dbutils.widgets.get("age_group_strategy")
VALIDATE_SCHEMA = dbutils.widgets.get("validate_schema").lower() == "true"
MODE = dbutils.widgets.get("mode")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Feature Engineering And Prediction

# COMMAND ----------

import pandas as pd

from medrisk_health_analytics import FeatureEngineeringPipeline
from medrisk_health_analytics.boosting import predict_with_registered_model

df_raw = spark.table(INPUT_TABLE).toPandas()

protected_columns = [c for c in ID_COLUMNS + [TARGET_COLUMN] if c in df_raw.columns]
df_features = df_raw.drop(columns=protected_columns)

pipeline = FeatureEngineeringPipeline(
    age_group_strategy=AGE_GROUP_STRATEGY,
    validate_schema=VALIDATE_SCHEMA,
)
df_engineered = pipeline.transform(df_features)

df_scoring = pd.concat(
    [
        df_raw[[c for c in ID_COLUMNS if c in df_raw.columns]].reset_index(drop=True),
        df_engineered.reset_index(drop=True),
    ],
    axis=1,
)

predictions = predict_with_registered_model(
    model_uri=MODEL_URI,
    input_df=df_scoring,
)

print(f"input rows       : {len(df_raw):,}")
print(f"prediction rows  : {len(predictions):,}")
print(f"output table     : {OUTPUT_TABLE}")
display(predictions.head(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Predictions

# COMMAND ----------

(
    spark.createDataFrame(predictions)
    .write.format("delta")
    .mode(MODE)
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"Predictions written to {OUTPUT_TABLE}")
