# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Set Model Alias
# MAGIC
# MAGIC Objective: promote a registered model version by assigning a Unity Catalog
# MAGIC alias such as `Champion` or `Challenger`.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("registered_model_name", "workspace.my_schema.boosting_risk_model")
dbutils.widgets.text("model_version", "")
dbutils.widgets.text("alias", "Champion")

REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name")
MODEL_VERSION = dbutils.widgets.get("model_version")
MODEL_ALIAS = dbutils.widgets.get("alias")

if not MODEL_VERSION:
    MODEL_VERSION = dbutils.jobs.taskValues.get(
        taskKey="train_register_model",
        key="registered_model_version",
        default="",
    )

if not MODEL_VERSION:
    raise ValueError("Parameter 'model_version' is required or must be produced by training.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Assign Alias

# COMMAND ----------

from medrisk_features.mlflow import set_model_alias

set_model_alias(
    registered_model_name=REGISTERED_MODEL_NAME,
    version=MODEL_VERSION,
    alias=MODEL_ALIAS,
)

print(f"Alias @{MODEL_ALIAS} assigned to {REGISTERED_MODEL_NAME} version {MODEL_VERSION}")
