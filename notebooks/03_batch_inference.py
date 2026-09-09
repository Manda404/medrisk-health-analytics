# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Batch Inference
# MAGIC
# MAGIC Objective: read a scoring table, apply feature engineering, score with a
# MAGIC registered model, and save predictions to Unity Catalog.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from medrisk_health_analytics.settings import settings

# COMMAND ----------
# MAGIC %md
# MAGIC ## Feature Engineering And Prediction

# COMMAND ----------

from medrisk_health_analytics import (
    DatasetLoader,
    DatasetPreprocessor,
    FeatureEngineeringPipeline,
    ModelPredictor,
)

patient_data = DatasetLoader(spark=spark).load(settings.scoring_table)
patient_data = DatasetPreprocessor().transform(patient_data)

feature_pipeline = FeatureEngineeringPipeline(
    age_group_strategy=settings.age_group_strategy,
    validate_schema=settings.validate_schema,
)
scoring_data = feature_pipeline.transform(patient_data, id_columns=settings.id_columns)

predictor = ModelPredictor(settings.registered_model_uri)
predictions = predictor.predict(scoring_data)

print(f"input rows       : {len(patient_data):,}")
print(f"prediction rows  : {len(predictions):,}")
print(f"output table     : {settings.scoring_predictions_table}")
display(predictions.head(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Predictions

# COMMAND ----------

(
    spark.createDataFrame(predictions)
    .write.format("delta")
    .mode(settings.write_mode)
    .option("overwriteSchema", "true")
    .saveAsTable(settings.scoring_predictions_table)
)

print(f"Predictions written to {settings.scoring_predictions_table}")
