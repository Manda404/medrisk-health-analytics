# Databricks notebook source
# MAGIC %md
# MAGIC # medrisk-health-analytics — Boosting MLOps Pipeline (XGBoost / CatBoost / LightGBM)
# MAGIC
# MAGIC **End-to-end notebook:** data loading → feature engineering → boosting training → MLflow → `BoostingPyFuncModel` → inference
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 1    | Install the package |
# MAGIC | 2    | Configuration |
# MAGIC | 3    | Load data from Unity Catalog |
# MAGIC | 4    | Feature engineering (medrisk pipeline) |
# MAGIC | 5    | Train boosting model with `train_boosting_model()` |
# MAGIC | 6    | Inspect MLflow run + metrics |
# MAGIC | 7    | Assign `@Champion` alias in Unity Catalog Registry |
# MAGIC | 8    | Batch inference with `predict_with_registered_model()` |
# MAGIC | 9    | Write results back to Unity Catalog |

# COMMAND ----------
# MAGIC %md ## 1. Install the package

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/Manda404/medrisk-health-analytics.git" xgboost catboost lightgbm mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## 2. Configuration — modify before running

# COMMAND ----------

# ── Unity Catalog ─────────────────────────────────────────────────────────────
CATALOG       = "workspace"
SCHEMA        = "my_schema"
TRAIN_TABLE   = f"{CATALOG}.{SCHEMA}.patient_training_data"
SCORING_TABLE = f"{CATALOG}.{SCHEMA}.patient_scoring_data"
OUTPUT_TABLE  = f"{CATALOG}.{SCHEMA}.boosting_predictions"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_TYPE    = "xgboost"   # "xgboost" | "catboost" | "lightgbm"
TARGET_COLUMN = "TARGET"
ID_COLUMNS    = ["client_id", "pers_id"]

# ── MLflow ────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME       = f"/Shared/experiments/{SCHEMA}/boosting"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.boosting_risk_model"
MODEL_ALIAS           = "Champion"

# ── Feature engineering ───────────────────────────────────────────────────────
AGE_GROUP_STRATEGY = "detailed"

# COMMAND ----------
# MAGIC %md ## 3. Load training data from Unity Catalog

# COMMAND ----------

import numpy as np
import pandas as pd

# ── Option A: real Unity Catalog table ────────────────────────────────────────
# from medrisk_health_analytics.boosting import read_from_delta
# df_spark = spark.table(TRAIN_TABLE)
# df = df_spark.toPandas()   # or use to_pandas() for automatic warnings

# ── Option B: synthetic data for demonstration ────────────────────────────────
np.random.seed(42)
N = 1000

df = pd.DataFrame({
    # Identifiers (kept in output, not used as features)
    "client_id": [f"C{i:05d}" for i in range(N)],
    "pers_id":   [f"P{i:05d}" for i in range(N)],

    # Demographics
    "Age":          np.random.randint(25, 80, N),
    "gender":       np.random.choice(["Male", "Female"], N),
    "income_level": np.random.choice(["Low", "Lower-Middle", "Middle", "High"], N),
    "education_level": np.random.choice(["No formal", "Highschool", "Bachelor", "Master"], N),
    "employment_status": np.random.choice(["Employed", "Retired", "Unemployed"], N),

    # Clinical biomarkers
    "glucose_fasting":   np.random.normal(105, 25, N).clip(60, 280),
    "hba1c":             np.random.normal(6.0,  1.0, N).clip(4.0, 12.0),
    "bmi":               np.random.normal(27,   5,   N).clip(16, 50),
    "systolic_bp":       np.random.normal(125, 18, N).clip(90, 200),
    "diastolic_bp":      np.random.normal(80,  10, N).clip(60, 120),
    "triglycerides":     np.random.normal(160, 60, N).clip(50, 500),
    "hdl_cholesterol":   np.random.normal(45,  12, N).clip(20, 100),
    "ldl_cholesterol":   np.random.normal(130, 35, N).clip(40, 300),
    "cholesterol_total": np.random.normal(200, 40, N).clip(100, 400),
    "insulin_level":     np.random.normal(15,   8, N).clip(2, 80),
    "glucose_postprandial": np.random.normal(145, 40, N).clip(80, 350),

    # Lifestyle
    "physical_activity_minutes_per_week": np.random.choice([0, 60, 150, 300], N),
    "screen_time_hours_per_day": np.random.uniform(1, 12, N),
    "sleep_hours_per_day":       np.random.uniform(4, 10, N),
    "diet_score":                np.random.randint(1, 11, N),
    "alcohol_consumption_per_week": np.random.choice([0, 1, 2, 4, 7], N),
    "smoking_status": np.random.choice(["Never", "Former", "Current"], N),
})

# Synthetic binary target (diabetes risk flag)
risk_score = (
    (df["glucose_fasting"] > 125).astype(int)
    + (df["bmi"] > 30).astype(int)
    + (df["hba1c"] > 6.5).astype(int)
    + (df["Age"] > 50).astype(int)
)
df["TARGET"] = (risk_score >= 2).astype(int)

print(f"Dataset shape : {df.shape}")
print(f"Target balance: {df['TARGET'].value_counts(normalize=True).to_dict()}")
display(df.head(3))

# COMMAND ----------
# MAGIC %md ## 4. Feature engineering with medrisk-health-analytics pipeline

# COMMAND ----------

from medrisk_health_analytics import FeatureEngineeringPipeline, SchemaValidationError

# Drop id columns and target before passing to the pipeline
df_for_engineering = df.drop(columns=ID_COLUMNS + [TARGET_COLUMN])

pipeline = FeatureEngineeringPipeline(
    age_group_strategy=AGE_GROUP_STRATEGY,
    validate_schema=True,
)

try:
    df_engineered = pipeline.transform(df_for_engineering)
    print(f"✅ Feature engineering: {df_for_engineering.shape[1]} → {df_engineered.shape[1]} columns")
except SchemaValidationError as e:
    print(f"❌ Schema error: {e}")
    raise

# Reattach identifiers and target
df_full = pd.concat([
    df[ID_COLUMNS].reset_index(drop=True),
    df_engineered.reset_index(drop=True),
    df[[TARGET_COLUMN]].reset_index(drop=True),
], axis=1)

print(f"Final dataset shape: {df_full.shape}")
display(df_full.head(3))

# COMMAND ----------
# MAGIC %md ## 5. Train the boosting model with full MLflow logging
# MAGIC
# MAGIC `train_boosting_model()` handles internally:
# MAGIC - Stratified train/validation/test split
# MAGIC - TabularPreprocessor (imputation + ordinal encoding)
# MAGIC - XGBoost / CatBoost / LightGBM training
# MAGIC - Evaluation on all three splits
# MAGIC - MLflow logging (params, metrics, artifacts, pyfunc model)
# MAGIC - Model Registry registration (Unity Catalog)

# COMMAND ----------

from medrisk_health_analytics.boosting import TrainingConfig, train_boosting_model

config = TrainingConfig(
    # Data
    target_column=TARGET_COLUMN,
    id_columns=ID_COLUMNS,
    exclude_columns=[],

    # Model
    model_type=MODEL_TYPE,
    model_params={
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },

    # Split
    test_size=0.2,
    validation_size=0.2,
    random_state=42,

    # MLflow
    experiment_name=EXPERIMENT_NAME,
    registered_model_name=REGISTERED_MODEL_NAME,
    run_name=f"medrisk-{MODEL_TYPE}-v1",

    # Priority thresholds
    priority_high_threshold=0.8,
    priority_medium_threshold=0.5,
)

result = train_boosting_model(df=df_full, config=config)

print("✅ Training complete!")
print(f"   run_id             : {result.run_id}")
print(f"   model_uri          : {result.model_uri}")
print(f"   registered_version : {result.registered_model_version}")
print(f"\n📊 Key metrics:")
for k in ["test_roc_auc", "test_f1", "test_avg_precision", "test_accuracy"]:
    if k in result.metrics:
        print(f"   {k:<25} : {result.metrics[k]:.4f}")

# COMMAND ----------
# MAGIC %md ## 6. Inspect MLflow metrics across all splits

# COMMAND ----------

import pandas as pd

# Display all metrics in a readable table
metrics_df = (
    pd.DataFrame(result.metrics.items(), columns=["metric", "value"])
    .assign(split=lambda d: d["metric"].str.split("_").str[0])
    .assign(name=lambda d: d["metric"].str.split("_", 1).str[1])
    .pivot(index="name", columns="split", values="value")
)

display(metrics_df.style.format("{:.4f}").highlight_max(axis=1, color="lightgreen"))

# COMMAND ----------
# MAGIC %md ## 7. Assign @Champion alias in Unity Catalog

# COMMAND ----------

import mlflow

if result.registered_model_version:
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=result.registered_model_version,
    )
    print(f"✅ Alias '@{MODEL_ALIAS}' assigned to version {result.registered_model_version}")

# COMMAND ----------
# MAGIC %md ## 8. Batch inference with predict_with_registered_model()
# MAGIC
# MAGIC The `BoostingPyFuncModel` loaded from the registry applies automatically:
# MAGIC 1. Feature selection (only the trained features)
# MAGIC 2. Preprocessor (imputation + encoding)
# MAGIC 3. XGBoost / CatBoost prediction
# MAGIC 4. Probability → priority mapping (HIGH / MEDIUM / LOW)
# MAGIC 5. Output formatting

# COMMAND ----------

from medrisk_health_analytics.boosting import predict_with_registered_model

# Simulate new unseen data (same raw format, no feature engineering needed)
df_scoring = df.sample(100, random_state=99).drop(columns=[TARGET_COLUMN])

# Apply medrisk feature engineering to scoring data
df_scoring_features = pipeline.transform(df_scoring.drop(columns=ID_COLUMNS))
df_scoring_full = pd.concat([
    df_scoring[ID_COLUMNS].reset_index(drop=True),
    df_scoring_features.reset_index(drop=True),
], axis=1)

# Inference using the @Champion model
model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

predictions = predict_with_registered_model(
    model_uri=model_uri,
    input_df=df_scoring_full,
)

print(f"✅ Inference complete — {len(predictions)} predictions")
print(f"\nPriority distribution:")
print(predictions["priority"].value_counts().to_string())

display(predictions.head(10))

# COMMAND ----------
# MAGIC %md ## 9. Write predictions back to Unity Catalog

# COMMAND ----------

# Uncomment to save to Delta table
#
# predict_with_registered_model(
#     model_uri=model_uri,
#     input_df=df_scoring_full,
#     output_table=OUTPUT_TABLE,
#     mode="overwrite",
# )
# print(f"✅ Predictions saved to {OUTPUT_TABLE}")
#
# display(spark.table(OUTPUT_TABLE).limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Component | What was used |
# MAGIC |-----------|---------------|
# MAGIC | Feature engineering | `FeatureEngineeringPipeline` (medrisk-health-analytics) |
# MAGIC | Model | `BoostingModelFactory.create("xgboost")` |
# MAGIC | Preprocessing | `TabularPreprocessor` (median imputation + ordinal encoding) |
# MAGIC | PyfuncModel | `BoostingPyFuncModel(mlflow.pyfunc.PythonModel)` |
# MAGIC | Tracking | `mlflow.log_params / log_metrics / log_model` |
# MAGIC | Registry | Unity Catalog `workspace.schema.model@Champion` |
# MAGIC | Inference | `predict_with_registered_model()` → client_id, pers_id, probability, priority |
