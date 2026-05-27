# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Feature Engineering Table
# MAGIC
# MAGIC Objective: apply `FeatureEngineeringPipeline` to one Unity Catalog table
# MAGIC and persist the enriched dataset to another Delta table.
# MAGIC
# MAGIC Run this notebook once for the train split and once for the test split.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("input_table", "workspace.my_schema.patient_train_raw")
dbutils.widgets.text("output_table", "workspace.my_schema.patient_train_features")
dbutils.widgets.text("target_column", "diagnosed_diabetes")
dbutils.widgets.text("id_columns", "")
dbutils.widgets.dropdown("age_group_strategy", "detailed", ["detailed", "coarse"])
dbutils.widgets.dropdown("validate_schema", "true", ["true", "false"])
dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "append"])

INPUT_TABLE = dbutils.widgets.get("input_table")
OUTPUT_TABLE = dbutils.widgets.get("output_table")
TARGET_COLUMN = dbutils.widgets.get("target_column")
ID_COLUMNS = [c.strip() for c in dbutils.widgets.get("id_columns").split(",") if c.strip()]
AGE_GROUP_STRATEGY = dbutils.widgets.get("age_group_strategy")
VALIDATE_SCHEMA = dbutils.widgets.get("validate_schema").lower() == "true"
MODE = dbutils.widgets.get("mode")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load And Transform

# COMMAND ----------

import pandas as pd

from medrisk_features import FeatureEngineeringPipeline

df = spark.table(INPUT_TABLE).toPandas()

protected_columns = [c for c in ID_COLUMNS + [TARGET_COLUMN] if c in df.columns]
df_features = df.drop(columns=protected_columns)

pipeline = FeatureEngineeringPipeline(
    age_group_strategy=AGE_GROUP_STRATEGY,
    validate_schema=VALIDATE_SCHEMA,
)

df_engineered = pipeline.transform(df_features)
df_output = pd.concat(
    [
        df[[c for c in ID_COLUMNS if c in df.columns]].reset_index(drop=True),
        df_engineered.reset_index(drop=True),
        df[[TARGET_COLUMN]].reset_index(drop=True) if TARGET_COLUMN in df.columns else pd.DataFrame(),
    ],
    axis=1,
)

print(f"input shape  : {df.shape}")
print(f"output shape : {df_output.shape}")
print(f"output table : {OUTPUT_TABLE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Output Table

# COMMAND ----------

(
    spark.createDataFrame(df_output)
    .write.format("delta")
    .mode(MODE)
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

display(spark.table(OUTPUT_TABLE).limit(10))
