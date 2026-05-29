# Databricks notebook source
# MAGIC %md
# MAGIC # medrisk-health-analytics — Feature Engineering + MLflow Integration
# MAGIC
# MAGIC This notebook demonstrates the full workflow:
# MAGIC 1. Install the package from GitHub
# MAGIC 2. Load data from Unity Catalog (or generate a sample)
# MAGIC 3. Run the FeatureEngineeringPipeline
# MAGIC 4. Log the pipeline as a `mlflow.pyfunc.PythonModel` (MedRiskPyFuncModel)
# MAGIC 5. Register in the MLflow Model Registry (Unity Catalog)
# MAGIC 6. Load back and run batch inference

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Install the package

# COMMAND ----------

# MAGIC %pip install git+https://github.com/Manda404/medrisk-health-analytics.git

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Configuration — modify these parameters before running

# COMMAND ----------

# ── Data source ───────────────────────────────────────────────────────────────
CATALOG          = "workspace"
SCHEMA           = "my_schema"
TRAIN_TABLE      = "patient_training_data"     # Unity Catalog table with raw features
INFERENCE_TABLE  = "patient_scoring_data"      # Table to score at inference time
OUTPUT_TABLE     = "patient_scoring_results"   # Where to write predictions

# ── MLflow ────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME        = f"/Shared/experiments/{SCHEMA}/medrisk_pipeline"
REGISTERED_MODEL_NAME  = f"{CATALOG}.{SCHEMA}.medrisk_feature_pipeline"
MODEL_ALIAS            = "Champion"            # Unity Catalog alias

# ── Pipeline ──────────────────────────────────────────────────────────────────
AGE_GROUP_STRATEGY = "detailed"   # "detailed" | "coarse"
VALIDATE_SCHEMA    = True

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Load training data from Unity Catalog

# COMMAND ----------

import pandas as pd
import numpy as np

# Option A — load from Unity Catalog
# df_spark = spark.table(f"{CATALOG}.{SCHEMA}.{TRAIN_TABLE}")
# df = df_spark.toPandas()

# Option B — synthetic sample for demonstration
np.random.seed(42)
N = 500

df = pd.DataFrame({
    "Age":                              np.random.randint(25, 80, N),
    "gender":                           np.random.choice(["Male", "Female", "Other"], N),
    "income_level":                     np.random.choice(["Low", "Lower-Middle", "Middle", "High"], N),
    "education_level":                  np.random.choice(["No formal", "Highschool", "Bachelor", "Master"], N),
    "glucose_fasting":                  np.random.normal(105, 25, N).clip(60, 280),
    "hba1c":                            np.random.normal(6.0, 1.0, N).clip(4.0, 12.0),
    "bmi":                              np.random.normal(27, 5, N).clip(16, 50),
    "systolic_bp":                      np.random.normal(125, 18, N).clip(90, 200),
    "diastolic_bp":                     np.random.normal(80, 10, N).clip(60, 120),
    "triglycerides":                    np.random.normal(160, 60, N).clip(50, 500),
    "hdl_cholesterol":                  np.random.normal(45, 12, N).clip(20, 100),
    "ldl_cholesterol":                  np.random.normal(130, 35, N).clip(40, 300),
    "cholesterol_total":                np.random.normal(200, 40, N).clip(100, 400),
    "insulin_level":                    np.random.normal(15, 8, N).clip(2, 80),
    "glucose_postprandial":             np.random.normal(145, 40, N).clip(80, 350),
    "physical_activity_minutes_per_week": np.random.choice([0, 60, 150, 300, 450], N),
    "screen_time_hours_per_day":        np.random.uniform(1, 12, N),
    "sleep_hours_per_day":              np.random.uniform(4, 10, N),
    "diet_score":                       np.random.randint(1, 11, N),
    "alcohol_consumption_per_week":     np.random.choice([0, 1, 2, 4, 7, 14], N),
    "smoking_status":                   np.random.choice(["Never", "Former", "Current"], N),
    "employment_status":                np.random.choice(["Employed", "Retired", "Unemployed"], N),
})

print(f"Dataset shape: {df.shape}")
display(df.head(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Build and run the FeatureEngineeringPipeline

# COMMAND ----------

from medrisk_health_analytics import FeatureEngineeringPipeline, SchemaValidationError

pipeline = FeatureEngineeringPipeline(
    age_group_strategy=AGE_GROUP_STRATEGY,
    validate_schema=VALIDATE_SCHEMA,
)

try:
    df_enriched = pipeline.transform(df)
    print(f"✅ Pipeline completed — {df.shape[1]} → {df_enriched.shape[1]} columns")
    print(f"\nNew features added:")
    new_cols = [c for c in df_enriched.columns if c not in df.columns]
    for col in new_cols:
        print(f"  + {col}")
except SchemaValidationError as e:
    print(f"❌ Schema validation failed: {e}")
    raise

display(df_enriched.head(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Log the pipeline to MLflow as a `PythonModel` (MedRiskPyFuncModel)
# MAGIC
# MAGIC The `log_pipeline()` function:
# MAGIC - Serializes the pipeline (`pipeline.pkl`)
# MAGIC - Saves the configuration (`config.json`)
# MAGIC - Saves the output feature names (`feature_names.json`)
# MAGIC - Logs everything inside a `mlflow.pyfunc.PythonModel` (MedRiskPyFuncModel)
# MAGIC - Optionally registers in the Unity Catalog Model Registry

# COMMAND ----------

from medrisk_health_analytics.mlflow import log_pipeline

result = log_pipeline(
    pipeline=pipeline,
    experiment_name=EXPERIMENT_NAME,
    artifact_path="medrisk_pipeline",
    df_sample=df.head(20),                        # used for signature + input example
    registered_model_name=REGISTERED_MODEL_NAME,  # remove to skip registry
    tags={
        "team": "data-science",
        "domain": "healthcare",
        "use_case": "diabetes_risk",
    },
    run_name="medrisk-feature-pipeline-v1",
)

print(f"✅ Model logged successfully")
print(f"   run_id              : {result.run_id}")
print(f"   model_uri           : {result.model_uri}")
print(f"   registered_model    : {result.registered_model_name}")
print(f"   registered_version  : {result.registered_model_version}")
print(f"   output features     : {len(result.feature_names)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Assign a Unity Catalog alias ('Champion')

# COMMAND ----------

from medrisk_health_analytics.mlflow import set_model_alias

if result.registered_model_version:
    set_model_alias(
        registered_model_name=REGISTERED_MODEL_NAME,
        version=result.registered_model_version,
        alias=MODEL_ALIAS,
    )
    print(f"✅ Alias '{MODEL_ALIAS}' assigned to version {result.registered_model_version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Load the registered model and run batch inference
# MAGIC
# MAGIC The loaded model is a `MedRiskPyFuncModel` — calling `.predict(df)` applies
# MAGIC the complete feature engineering pipeline automatically.

# COMMAND ----------

from medrisk_health_analytics.mlflow import load_pipeline

# Load by alias (Unity Catalog)
model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

# Alternative: load by run URI (no registry needed)
# model_uri = result.model_uri

loaded_model = load_pipeline(model_uri)
print(f"✅ Model loaded from: {model_uri}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Run inference on new (unseen) data

# COMMAND ----------

# Simulate new scoring data (raw — no engineered features)
df_new = df.sample(50, random_state=99).drop(
    columns=[c for c in df_enriched.columns if c not in df.columns],
    errors="ignore",
)
print(f"Scoring data shape: {df_new.shape}")

# Apply the full pipeline via the pyfunc model
df_predictions = loaded_model.predict(df_new)

print(f"✅ Inference complete — output shape: {df_predictions.shape}")
display(df_predictions.head(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ### (Optional) Write inference results back to Unity Catalog

# COMMAND ----------

# Uncomment to save results to a Delta table
#
# df_predictions_spark = spark.createDataFrame(df_predictions)
# (
#     df_predictions_spark
#     .write
#     .format("delta")
#     .mode("overwrite")
#     .option("mergeSchema", "true")
#     .saveAsTable(f"{CATALOG}.{SCHEMA}.{OUTPUT_TABLE}")
# )
# print(f"✅ Results written to {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | What happened |
# MAGIC |------|---------------|
# MAGIC | Pipeline | `FeatureEngineeringPipeline` enriched raw data with 20+ clinical features |
# MAGIC | MLflow | `MedRiskPyFuncModel` logged as pyfunc with signature + input example |
# MAGIC | Registry | Model registered in Unity Catalog as `{REGISTERED_MODEL_NAME}` |
# MAGIC | Alias | Version tagged `@Champion` for production use |
# MAGIC | Inference | Raw data → enriched features via `loaded_model.predict()` |
