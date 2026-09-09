# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Train, Validate And Promote
# MAGIC
# MAGIC Train and register one candidate, evaluate that exact model on the
# MAGIC holdout table, then assign the configured alias only if quality gates pass.

# COMMAND ----------

import pandas as pd

from medrisk_health_analytics import ModelTrainer
from medrisk_health_analytics.boosting import evaluate_registered_model_on_holdout
from medrisk_health_analytics.mlflow import set_model_alias
from medrisk_health_analytics.settings import settings

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train and register candidate

# COMMAND ----------

training_data = spark.table(settings.train_table).toPandas()
trainer = ModelTrainer(
    target_column=settings.target_column,
    id_columns=list(settings.id_columns),
    model_type=settings.model_type,
    test_size=settings.test_size,
    validation_size=settings.validation_size,
    random_state=settings.random_state,
    experiment_name=settings.mlflow_experiment_name,
    registered_model_name=settings.registered_model_name,
    run_name=settings.run_name,
    priority_high_threshold=0.8,
    priority_medium_threshold=0.5,
    use_internal_test_split=False,
    training_data_source=settings.train_table,
    environment=settings.environment,
)
training_result = trainer.fit(training_data)

if not training_result.registered_model_version:
    raise RuntimeError("MLflow did not register the candidate model.")

print(f"Candidate URI: {training_result.model_uri}")
print(f"Candidate version: {training_result.registered_model_version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Evaluate candidate on holdout data

# COMMAND ----------

evaluation = evaluate_registered_model_on_holdout(
    model_uri=training_result.model_uri,
    input_table=settings.test_table,
    target_column=settings.target_column,
    output_table=(
        settings.holdout_predictions_table if settings.write_holdout_predictions else None
    ),
    mode=settings.write_mode,
    spark=spark,
)

quality_gates = {
    "holdout_roc_auc": settings.min_roc_auc,
    "holdout_recall": settings.min_recall,
}
failures = {
    metric: (evaluation.metrics[metric], minimum)
    for metric, minimum in quality_gates.items()
    if evaluation.metrics[metric] < minimum
}
if failures:
    details = ", ".join(
        f"{metric}={actual:.4f} < {minimum:.4f}"
        for metric, (actual, minimum) in failures.items()
    )
    raise RuntimeError(f"Candidate rejected: {details}")

display(pd.DataFrame(evaluation.metrics.items(), columns=["metric", "value"]))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Promote accepted version

# COMMAND ----------

set_model_alias(
    registered_model_name=settings.registered_model_name,
    version=training_result.registered_model_version,
    alias=settings.model_alias,
)

print(
    f"Promoted {settings.registered_model_name} version "
    f"{training_result.registered_model_version} as @{settings.model_alias}"
)
