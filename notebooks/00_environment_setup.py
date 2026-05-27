# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Environment Setup
# MAGIC
# MAGIC Objective: install `medrisk-features` and restart Python on Databricks.
# MAGIC
# MAGIC Run this notebook as the first task when the package is not installed as a
# MAGIC cluster library.

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/Manda404/medrisk-features.git#egg=medrisk-features[boosting]" xgboost catboost lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import medrisk_features

print(f"medrisk_features version: {medrisk_features.__version__}")
