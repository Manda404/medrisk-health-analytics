# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Monitor Data Drift
# MAGIC
# MAGIC Compare the model input population with the training reference, persist
# MAGIC feature-level PSI metrics, and optionally stop the job on drift.

# COMMAND ----------

from datetime import UTC, datetime

from medrisk_health_analytics import DataDriftMonitor
from medrisk_health_analytics.settings import settings

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load reference and current populations

# COMMAND ----------

reference = spark.table(settings.train_table).drop(settings.target_column).toPandas()
current = spark.table(settings.scoring_table).toPandas()

monitor = DataDriftMonitor(psi_threshold=settings.drift_psi_threshold)
report = monitor.compare(reference, current)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Persist monitoring evidence

# COMMAND ----------

metrics = report.metrics.copy()
metrics["environment"] = settings.environment
metrics["model_uri"] = settings.registered_model_uri
metrics["measured_at"] = datetime.now(UTC)

(
    spark.createDataFrame(metrics)
    .write.format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(settings.drift_metrics_table)
)

print(f"Drift metrics appended to {settings.drift_metrics_table}")
print(f"Drifted features: {list(report.drifted_features)}")
display(metrics.head(20))

if settings.fail_on_drift:
    report.raise_for_failure()
