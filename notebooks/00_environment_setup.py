# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Environment Setup
# MAGIC
# MAGIC Objective: install `medrisk-health-analytics` and restart Python on Databricks.
# MAGIC
# MAGIC Run this notebook as the first task when the package is not installed as a
# MAGIC cluster library.

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/Manda404/medrisk-health-analytics.git#egg=medrisk-health-analytics[boosting]" xgboost catboost lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import medrisk_health_analytics

print(f"medrisk_health_analytics version: {medrisk_health_analytics.__version__}")
