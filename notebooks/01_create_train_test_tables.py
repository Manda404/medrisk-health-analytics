# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Create Train/Test Tables
# MAGIC
# MAGIC Objective: read one raw Unity Catalog table, create a reproducible
# MAGIC stratified split, and save two Delta tables: train and test.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("source_table", "workspace.my_schema.patient_raw_data")
dbutils.widgets.text("train_table", "workspace.my_schema.patient_train_raw")
dbutils.widgets.text("test_table", "workspace.my_schema.patient_test_raw")
dbutils.widgets.text("target_column", "diagnosed_diabetes")
dbutils.widgets.text("test_size", "0.2")
dbutils.widgets.text("random_state", "42")
dbutils.widgets.dropdown("stratify", "true", ["true", "false"])
dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "append"])

SOURCE_TABLE = dbutils.widgets.get("source_table")
TRAIN_TABLE = dbutils.widgets.get("train_table")
TEST_TABLE = dbutils.widgets.get("test_table")
TARGET_COLUMN = dbutils.widgets.get("target_column")
TEST_SIZE = float(dbutils.widgets.get("test_size"))
RANDOM_STATE = int(dbutils.widgets.get("random_state"))
STRATIFY = dbutils.widgets.get("stratify").lower() == "true"
MODE = dbutils.widgets.get("mode")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Split And Persist

# COMMAND ----------

from medrisk_health_analytics.boosting import create_train_test_tables

result = create_train_test_tables(
    source_table=SOURCE_TABLE,
    train_table=TRAIN_TABLE,
    test_table=TEST_TABLE,
    target_column=TARGET_COLUMN,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=STRATIFY,
    mode=MODE,
    spark=spark,
)

print("Train/test split completed")
print(f"source_table : {result.source_table}")
print(f"train_table  : {result.train_table} ({result.train_count:,} rows)")
print(f"test_table   : {result.test_table} ({result.test_count:,} rows)")
print(f"train target : {result.train_target_distribution}")
print(f"test target  : {result.test_target_distribution}")

# COMMAND ----------

display(spark.table(TRAIN_TABLE).limit(10))
display(spark.table(TEST_TABLE).limit(10))
