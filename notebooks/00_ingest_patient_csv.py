# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Ingest Patient CSV
# MAGIC
# MAGIC Objective: read the original patient CSV from a path or Unity Catalog
# MAGIC volume and persist it as the raw Delta source table used by the pipeline.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("source_path", "/Volumes/workspace/my_schema/raw/patient.csv")
dbutils.widgets.text("source_table", "workspace.my_schema.patient_raw_data")
dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "append"])

SOURCE_PATH = dbutils.widgets.get("source_path")
SOURCE_TABLE = dbutils.widgets.get("source_table")
MODE = dbutils.widgets.get("mode")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Read CSV

# COMMAND ----------

df_raw = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(SOURCE_PATH)
)

print(f"source_path  : {SOURCE_PATH}")
print(f"source_table : {SOURCE_TABLE}")
print(f"rows         : {df_raw.count():,}")
print(f"columns      : {len(df_raw.columns)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Raw Source Table

# COMMAND ----------

(
    df_raw.write.format("delta")
    .mode(MODE)
    .option("overwriteSchema", "true")
    .saveAsTable(SOURCE_TABLE)
)

display(spark.table(SOURCE_TABLE).limit(10))
