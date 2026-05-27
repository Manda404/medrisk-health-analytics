# medrisk-features

![CI](https://github.com/Manda404/medrisk-features/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![Version](https://img.shields.io/badge/version-0.2.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Databricks](https://img.shields.io/badge/Databricks-compatible-red?logo=databricks)](https://databricks.com)
[![MLflow](https://img.shields.io/badge/MLflow-pyfunc-blue?logo=mlflow)](https://mlflow.org)

**medrisk-features** is a production-ready Python package for **clinical feature engineering and boosting MLOps on Databricks**.

It gives you three complementary layers that can be used independently or together:

| Layer | Entry point | What it does |
|-------|-------------|--------------|
| **Feature Engineering** | `FeatureEngineeringPipeline` | Clinically-grounded transformations — glucose, BMI, lipids, blood pressure, lifestyle — aligned with ADA / WHO / JNC / NCEP-ATP III guidelines |
| **Boosting MLOps** | `train_boosting_model` | End-to-end XGBoost / CatBoost / LightGBM pipeline with preprocessing, MLflow logging, Unity Catalog registration, and batch inference |
| **Explainability (SHAP)** | `BoostingShapExplainer` | SHAP-based feature importance with summary, bar, and waterfall plots — model-agnostic, Databricks-compatible |

---

## Table of Contents

1. [Why medrisk-features?](#1-why-medrisk-features)
2. [Installation](#2-installation)
3. [Quick Start — 5 minutes to your first enriched DataFrame](#3-quick-start)
4. [Feature Engineering Pipeline — deep dive](#4-feature-engineering-pipeline)
   - [Initialisation](#41-initialisation)
   - [Required and optional columns](#42-required-and-optional-columns)
   - [All engineered features](#43-all-engineered-features)
   - [Error handling](#44-error-handling)
5. [MLflow Integration — logging the feature pipeline](#5-mlflow-integration)
6. [Boosting MLOps Layer](#6-boosting-mlops-layer)
   - [TrainingConfig](#61-trainingconfig)
   - [Training](#62-training)
   - [Holdout evaluation](#63-holdout-evaluation)
   - [Batch inference](#64-batch-inference)
7. [SHAP Explainability](#7-shap-explainability)
   - [Quick start](#71-quick-start)
   - [Summary plot (beeswarm)](#72-summary-plot)
   - [Bar plot](#73-bar-plot)
   - [Waterfall plot](#74-waterfall-plot)
   - [SHAP DataFrame](#75-shap-dataframe)
8. [Complete Databricks Walkthrough](#8-complete-databricks-walkthrough)
9. [Project Structure](#9-project-structure)
10. [Local Development](#10-local-development)
11. [Roadmap](#11-roadmap)
12. [Author & License](#12-author--license)

---

## 1. Why medrisk-features?

Building healthcare ML models requires turning raw patient measurements into meaningful, clinically-interpretable features. Without a structured approach, teams end up reimplementing the same glucose/BMI/lipid logic across projects, often with subtle threshold errors and no traceability to the underlying guidelines.

**medrisk-features solves this by providing:**

- **Clinically accurate thresholds** hard-coded against ADA 2023, WHO, JNC 7, and NCEP-ATP III — no more guessing where "normal" glucose ends.
- **A single `pipeline.transform(df)` call** that takes raw measurements and returns 40+ engineered features.
- **A battle-tested MLOps layer** that trains, logs, and deploys XGBoost / CatBoost / LightGBM models with MLflow and Unity Catalog in one function call.
- **No silent failures** — explicit errors with actionable messages when required columns are missing.

---

## 2. Installation

### On Databricks (in a notebook cell)

```python
# Core feature engineering only
%pip install git+https://github.com/Manda404/medrisk-features.git

# With MLflow tracking support
%pip install "git+https://github.com/Manda404/medrisk-features.git#egg=medrisk-features[mlflow]"

# Full boosting stack — recommended for end-to-end workflows
%pip install "git+https://github.com/Manda404/medrisk-features.git#egg=medrisk-features[boosting]"

# Always restart Python after %pip install in Databricks
dbutils.library.restartPython()
```

### With pip

```bash
# Core package (no ML libraries)
pip install git+https://github.com/Manda404/medrisk-features.git

# With optional extras
pip install "medrisk-features[mlflow]"      # + MLflow tracking
pip install "medrisk-features[xgboost]"     # + XGBoost
pip install "medrisk-features[catboost]"    # + CatBoost
pip install "medrisk-features[lightgbm]"    # + LightGBM
pip install "medrisk-features[boosting]"    # + all 3 boosting libs + sklearn + mlflow
pip install "medrisk-features[all]"         # everything
```

> **Which extra should I use?**
> - Just enriching features for another model → bare install
> - Logging the pipeline to MLflow → `[mlflow]`
> - Training + deploying a boosting model → `[boosting]`

### With Poetry (for library development)

```bash
git clone https://github.com/Manda404/medrisk-features.git
cd medrisk-features
poetry install
```

---

## 3. Quick Start

Let's go from raw patient data to an enriched DataFrame in under 5 minutes.

### Step 1 — Prepare your DataFrame

Your DataFrame needs at minimum these three columns:

| Column | Type | Description |
|--------|------|-------------|
| `Age` | numeric | Patient age in years |
| `glucose_fasting` | numeric | Fasting glucose in mg/dL |
| `bmi` | numeric | Body mass index in kg/m² |

```python
import pandas as pd

df = pd.DataFrame({
    "Age":              [45, 62, 33],
    "glucose_fasting":  [105, 140, 88],
    "bmi":              [28.5, 34.2, 22.1],
    "systolic_bp":      [125, 145, 110],
    "diastolic_bp":     [82,  92,  70],
    "hba1c":            [5.9, 7.1, 5.2],
    "triglycerides":    [180, 220, 95],
    "hdl_cholesterol":  [38,  35,  55],
})
```

### Step 2 — Run the pipeline

```python
from medrisk_features import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline(
    age_group_strategy="detailed",  # "detailed" (fine bands) or "coarse" (3 groups)
    validate_schema=True,           # raises an explicit error if columns are missing
)

df_enriched = pipeline.transform(df)

print(f"Raw columns:      {df.shape[1]}")
print(f"Enriched columns: {df_enriched.shape[1]}")
print(df_enriched[["bmi_category", "glucose_category", "bp_category", "metabolic_syndrome_flag"]].head())
```

Output:

```
Raw columns:      8
Enriched columns: 37

  bmi_category glucose_category bp_category  metabolic_syndrome_flag
0   Overweight      Pre-Diabetes  Hypertension                     True
1        Obese          Diabetes  Hypertension                     True
2       Normal           Normal        Normal                    False
```

### Step 3 — Use the enriched features for ML

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

y = pd.Series([1, 1, 0])  # target label

X_train, X_test, y_train, y_test = train_test_split(
    df_enriched.select_dtypes("number"), y, test_size=0.2, random_state=42
)

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
```

That's it. The pipeline produces numeric and categorical features ready for any sklearn-compatible model. For production training with XGBoost / CatBoost / LightGBM, continue to [Section 6](#6-boosting-mlops-layer).

---

## 4. Feature Engineering Pipeline — deep dive

### 4.1 Initialisation

```python
from medrisk_features import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline(
    age_group_strategy="detailed",  # "detailed" | "coarse"
    validate_schema=True,
)
```

`age_group_strategy` controls how the `Age` column is binned:

| Strategy | Bands produced |
|----------|---------------|
| `"detailed"` | `<30`, `30–39`, `40–49`, `50–59`, `60–69`, `70–79`, `80+` |
| `"coarse"` | `Young (<40)`, `Adult (40–64)`, `Senior (65+)` |

The `transform(df)` method is **pure** — it never modifies the input DataFrame.

### 4.2 Required and optional columns

The pipeline validates your DataFrame before doing any work. Missing required columns raise a `MissingRequiredColumnError` immediately.

**Required (always):**

| Column | Unit |
|--------|------|
| `Age` | years |
| `glucose_fasting` | mg/dL |
| `bmi` | kg/m² |

**Optional — each column unlocks additional features:**

| Column(s) | Features unlocked |
|-----------|------------------|
| `hba1c` | `hba1c_category` |
| `insulin_level` | `homa_ir`, `insulin_resistance_flag` |
| `systolic_bp`, `diastolic_bp` | `bp_category`, `blood_pressure_ratio` |
| `triglycerides` | part of `dyslipidemia_flag`, `metabolic_syndrome_flag` |
| `hdl_cholesterol` | `dyslipidemia_flag`, `lipid_ratio_hdl_ldl`, part of `metabolic_syndrome_flag` |
| `ldl_cholesterol` | `lipid_ratio_hdl_ldl`, `cholesterol_hdl_ratio` |
| `physical_activity_minutes_per_week` | `physical_activity_adequate`, `sedentary_risk` |
| `screen_time_hours_per_day` | `screen_sleep_imbalance`, `sedentary_risk` |
| `sleep_hours_per_day` | `sleep_efficiency`, `screen_sleep_imbalance` |
| `diet_score` | part of `lifestyle_score` |
| `smoking_status` | part of `lifestyle_score` |
| `alcohol_consumption_per_week` | part of `lifestyle_score` |

### 4.3 All engineered features

#### Demographics
| Feature | Description | Guideline |
|---------|-------------|-----------|
| `age_group` | Age band based on chosen strategy | — |
| `age_squared` | Non-linear aging effect (Age²) | — |
| `socioeconomic_vulnerability_flag` | Low income + low education indicator | — |

#### Medical
| Feature | Description | Guideline |
|---------|-------------|-----------|
| `glucose_category` | `Normal` / `Pre-Diabetes` / `Diabetes` | ADA 2023 (≤99 / 100–125 / ≥126 mg/dL) |
| `hba1c_category` | Glycemic control stratification | ADA 2023 (<5.7 / 5.7–6.4 / ≥6.5 %) |
| `bmi_category` | `Underweight` / `Normal` / `Overweight` / `Obese` | WHO (<18.5 / 18.5–25 / 25–30 / ≥30) |
| `bp_category` | `Normal` / `Pre-Hypertension` / `Hypertension` | JNC 7 |
| `homa_ir` | Insulin resistance index = (glucose × insulin) / 405 | — |
| `insulin_resistance_flag` | `homa_ir > 2.5` | Clinical cutoff |
| `metabolic_syndrome_flag` | ≥ 3 of the 5 ATP-III criteria | NCEP-ATP III |

#### Clinical interactions
| Feature | Description |
|---------|-------------|
| `bmi_glucose_interaction` | BMI × fasting glucose synergy score |
| `glucose_variability` | Postprandial − fasting glucose excursion |
| `lipid_ratio_hdl_ldl` | HDL / LDL atherogenic ratio |
| `cholesterol_hdl_ratio` | Total cholesterol / HDL ratio |

#### Metabolic
| Feature | Description |
|---------|-------------|
| `glycemic_load` | Glucose × BMI burden proxy |
| `dyslipidemia_flag` | Triglycerides ≥ 150 mg/dL **or** HDL < 40 mg/dL |
| `cardiometabolic_burden` | Composite score 0–5 (count of risk factors) |
| `blood_pressure_ratio` | Systolic / diastolic ratio |

#### Behavioral
| Feature | Description |
|---------|-------------|
| `physical_activity_adequate` | Activity / WHO 150 min/week ratio |
| `screen_sleep_imbalance` | Screen time / sleep hours ratio |
| `sedentary_risk` | High screen time + activity below WHO threshold |

#### Lifestyle
| Feature | Description |
|---------|-------------|
| `lifestyle_score` | Global health score 0–10 (5 binary criteria × 2) |
| `sleep_efficiency` | Sleep hours / (screen time + 1) |

### 4.4 Error handling

```python
from medrisk_features import FeatureEngineeringPipeline, MissingRequiredColumnError

pipeline = FeatureEngineeringPipeline(validate_schema=True)

try:
    df_out = pipeline.transform(df)
except MissingRequiredColumnError as e:
    # e.missing_columns → list of column names that are missing
    print(f"Missing: {e.missing_columns}")
    # Output example: Missing: ['glucose_fasting', 'bmi']
```

All exceptions in the package inherit from `MedRiskError` and are importable from the top-level module:

```python
from medrisk_features import (
    MedRiskError,              # base class for all package exceptions
    SchemaValidationError,     # schema validation failed
    MissingRequiredColumnError,# required column absent from input DataFrame
    FeatureEngineeringError,   # error during feature computation
    InvalidConfigurationError, # bad configuration value
)
```

---

## 5. MLflow Integration

The feature engineering pipeline itself can be versioned in MLflow, registered in the Unity Catalog Model Registry, and reloaded for reproducible inference.

### Log the pipeline after fitting

```python
from medrisk_features import FeatureEngineeringPipeline
from medrisk_features.mlflow import log_pipeline, load_pipeline, set_model_alias

# 1. Build and run the pipeline
pipeline = FeatureEngineeringPipeline(validate_schema=True)
df_enriched = pipeline.transform(df_train)

# 2. Log the pipeline to MLflow as a pyfunc model
result = log_pipeline(
    pipeline=pipeline,
    experiment_name="/Shared/experiments/medrisk",
    df_sample=df_train.head(20),           # used to infer the MLflow signature
    registered_model_name="workspace.analytics.medrisk_pipeline",
    tags={"team": "data-science", "domain": "healthcare"},
    run_name="feature-pipeline-v1",
)

print(result.run_id)          # → abc123
print(result.model_uri)       # → runs:/abc123/medrisk_pipeline
print(result.feature_names)   # → ['age_group', 'bmi_category', ...]
```

### Promote to production with an alias

```python
# Assign @Champion to the version you just registered
set_model_alias(
    registered_model_name="workspace.analytics.medrisk_pipeline",
    version=result.registered_model_version,
    alias="Champion",
)
```

### Load and run inference

```python
# Load by alias (Unity Catalog style — recommended)
loaded = load_pipeline("models:/workspace.analytics.medrisk_pipeline@Champion")
df_enriched = loaded.predict(df_raw)

# Or load by version number
loaded = load_pipeline("models:/workspace.analytics.medrisk_pipeline/3")

# Or load by run URI (no registry)
loaded = load_pipeline(result.model_uri)
```

`load_pipeline()` returns an `mlflow.pyfunc.PyFuncModel`. Call `.predict(df)` to apply the full feature engineering pipeline to any new DataFrame with the same schema.

---

## 6. Boosting MLOps Layer

The `medrisk_features.boosting` subpackage provides a **one-call training pipeline** for XGBoost, CatBoost, and LightGBM, with built-in preprocessing, MLflow experiment tracking, model registration, and structured batch inference.

The architecture looks like this:

```
raw DataFrame
      │
      ▼
 auto_detect_column_types()
      │
      ▼
 TabularPreprocessor.fit_transform()   ← imputation + encoding
      │
      ▼
 BoostingModel.fit()                   ← XGBoost / CatBoost / LightGBM
      │
      ▼
 evaluate_splits()                     ← train / valid / test metrics
      │
      ▼
 MLflow: log params, metrics, artifacts, pyfunc model, registry
      │
      ▼
 TrainingResult (run_id, model_uri, metrics, feature_names, …)
```

### 6.1 TrainingConfig

`TrainingConfig` is the single object that controls the entire training run. All fields have sensible defaults.

```python
from medrisk_features.boosting import TrainingConfig, SplitStrategy, TaskType

config = TrainingConfig(
    # ── Target & identifiers ─────────────────────────────────────────────
    target_column="TARGET",           # name of the label column
    id_columns=["client_id", "pers_id"],  # kept in output, never used as features

    # ── Feature selection ────────────────────────────────────────────────
    exclude_columns=["date_enrollment"],  # drop before training (dates, free text…)
    # numeric_columns=None,           # auto-detected if not provided
    # categorical_columns=None,       # auto-detected if not provided

    # ── Model ────────────────────────────────────────────────────────────
    model_type="xgboost",             # "xgboost" | "catboost" | "lightgbm"
    task_type=TaskType.BINARY_CLASSIFICATION,
    model_params={                    # passed directly to the library
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    # ── Split ────────────────────────────────────────────────────────────
    split_strategy=SplitStrategy.STRATIFIED,
    test_size=0.2,
    validation_size=0.2,
    use_internal_test_split=True,     # set False if you provide a separate test table
    random_state=42,

    # ── MLflow ───────────────────────────────────────────────────────────
    experiment_name="/Shared/experiments/diabetes",
    registered_model_name="workspace.analytics.diabetes_model",
    run_name="xgboost-v1",
    artifact_path="boosting_model",

    # ── Output priority thresholds ───────────────────────────────────────
    priority_high_threshold=0.8,      # probability ≥ 0.8  → HIGH
    priority_medium_threshold=0.5,    # probability ≥ 0.5  → MEDIUM
)                                     # otherwise          → LOW
```

### 6.2 Training

```python
from medrisk_features.boosting import train_boosting_model

result = train_boosting_model(df=df_train, config=config)

# TrainingResult fields
print(result.run_id)                    # MLflow run identifier
print(result.model_uri)                 # runs:/<run_id>/boosting_model
print(result.registered_model_version)  # "1", "2", …

# Metrics logged for each split (train / valid / test)
print(result.metrics)
# {
#   "train_roc_auc": 0.94,
#   "valid_roc_auc": 0.89,
#   "test_roc_auc":  0.87,
#   "test_f1":       0.81,
#   "test_precision":0.83,
#   "test_recall":   0.79,
# }
```

> `train_boosting_model()` automatically handles: column type detection, stratified split, `TabularPreprocessor` fit/transform, model training, evaluation on all splits, MLflow params/metrics/artifacts logging, pyfunc model registration in the Unity Catalog.

**Switching models is one word:**

```python
config.model_type = "catboost"   # or "lightgbm"
result = train_boosting_model(df=df_train, config=config)
```

### 6.3 Holdout evaluation

If you kept a completely separate holdout table in Unity Catalog, evaluate the registered model on it without retraining:

```python
from medrisk_features.boosting import evaluate_registered_model_on_holdout

eval_result = evaluate_registered_model_on_holdout(
    model_uri="models:/workspace.analytics.diabetes_model@Champion",
    holdout_table="workspace.analytics.patient_holdout",
    target_column="TARGET",
    spark=spark,
)

print(eval_result.roc_auc)     # → 0.872
print(eval_result.f1)          # → 0.794
print(eval_result.metrics)     # full dict of all metrics
```

### 6.4 Batch inference

`predict_with_registered_model()` loads a registered model and returns a clean prediction DataFrame — Spark or Pandas input, both accepted.

```python
from medrisk_features.boosting import predict_with_registered_model

# From a Pandas DataFrame
predictions = predict_with_registered_model(
    model_uri="models:/workspace.analytics.diabetes_model@Champion",
    input_df=df_scoring,
)

# From a Unity Catalog table (Spark)
predictions = predict_with_registered_model(
    model_uri="models:/workspace.analytics.diabetes_model@Champion",
    input_table="workspace.analytics.patient_scoring",
)
```

**Output format:**

| Column | Description |
|--------|-------------|
| `client_id`, `pers_id` | Identifier columns (configured via `id_columns`) |
| `probability` | Positive-class probability, rounded to 6 decimals |
| `priority` | `HIGH` / `MEDIUM` / `LOW` based on configured thresholds |
| `prediction_date` | UTC timestamp of the inference run |
| `model_version` | Registered model version used |

---

## 7. SHAP Explainability

After training a model with `train_boosting_model()`, you often need to answer: *which features drive this model's predictions, and in which direction?* The `BoostingShapExplainer` answers this using SHAP (SHapley Additive exPlanations) — the gold standard for ML interpretability.

**Install the extra:**
```bash
pip install "medrisk-features[explainability]"
# or: pip install "medrisk-features[boosting]"  (already includes shap + matplotlib)
```

### 7.1 Quick start

```python
from medrisk_features.boosting import BoostingShapExplainer

# After training:
# result = train_boosting_model(df=df_full, config=config)
# The model and preprocessed training data are needed

explainer = BoostingShapExplainer(
    model=result.model,                   # fitted BaseBoostingModel
    feature_names=result.feature_names,  # list of feature column names
)

# Fit the explainer on the preprocessed training set
# (same DataFrame that went into the booster — after TabularPreprocessor)
explainer.fit(X_train_preprocessed)

# Quick ranking: top 10 features by mean |SHAP|
print(explainer.top_features(n=10))
# glucose_category        0.2341
# bmi_category            0.1823
# homa_ir                 0.1204
# age_group               0.0987
# ...
```

### 7.2 Summary plot

The **beeswarm summary plot** is the single most informative SHAP visualisation. Each dot is one patient. The x-axis shows the SHAP value (positive = pushes toward high risk; negative = toward low risk). Colour encodes the original feature value (red = high, blue = low).

```python
fig = explainer.plot_summary(
    max_display=20,                          # show top 20 features
    title="SHAP Feature Impact — Diabetes Risk Model",
    figsize=(10, 7),
    alpha=0.6,
)
display(fig)                                 # Databricks
fig.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
```

Reading the plot:
- A feature at the top has high mean |SHAP| — it strongly influences predictions.
- Red dots on the right → high values of this feature **increase** predicted risk.
- Blue dots on the right → low values of this feature **increase** predicted risk.

### 7.3 Bar plot

The **bar chart** ranks features purely by mean |SHAP| value — simpler than the beeswarm but easier to share with non-technical stakeholders.

```python
fig = explainer.plot_bar(
    max_display=15,
    title="Top Features by SHAP Importance",
    color="#2196F3",
)
display(fig)
```

Unlike the model's built-in `get_feature_importance()` (which returns Gini impurity or gain), mean |SHAP| is measured in the same unit as the model output, making it directly comparable across models and datasets.

### 7.4 Waterfall plot

The **waterfall plot** explains a single patient prediction: starting from the model's base value (average prediction across all training samples), each bar shows how much each feature pushed the final prediction up (red) or down (blue).

```python
# Explain patient at row 0 of the training set
fig = explainer.plot_waterfall(
    sample_idx=0,
    max_display=12,
)
display(fig)

# Explain a high-risk patient (find one first)
import numpy as np
high_risk_idx = np.argmax(model.predict_proba(X_train_preprocessed))
fig = explainer.plot_waterfall(sample_idx=int(high_risk_idx))
display(fig)
```

The waterfall plot answers: *"Why did this specific patient receive a HIGH risk score?"* — the ideal tool for clinical audits and individual explanations.

### 7.5 SHAP DataFrame

For custom analysis (correlation with outcomes, SHAP-based segmentation, etc.), export the SHAP values as a plain DataFrame:

```python
shap_df = explainer.get_shap_dataframe()
# Shape: (n_samples, n_features) — same index as X_train_preprocessed

# Which feature had the largest individual SHAP impact in the dataset?
print(shap_df.abs().max().sort_values(ascending=False).head(5))

# Distribution of glucose_category SHAP values
print(shap_df["glucose_category"].describe())

# Positive vs negative SHAP contributions for BMI
positive_bmi = (shap_df["bmi_category"] > 0).sum()
print(f"BMI increased risk for {positive_bmi}/{len(shap_df)} patients")
```

You can also access the full `ShapResult` object for programmatic use:

```python
result_obj = explainer.result

result_obj.shap_values        # np.ndarray shape (n_samples, n_features)
result_obj.expected_value     # float — model base value
result_obj.mean_abs_shap      # pd.Series — ranked importance
result_obj.X                  # pd.DataFrame — input used to compute SHAP values
result_obj.top_features(n=5)  # pd.Series — top 5 features
```

---

## 8. Complete Databricks Walkthrough

This section walks through a real end-to-end workflow: from a raw Unity Catalog table to production predictions written back to Delta.

### Cell 1 — Install

```python
%pip install "git+https://github.com/Manda404/medrisk-features.git#egg=medrisk-features[boosting]"
dbutils.library.restartPython()
```

### Cell 2 — Configuration (edit this cell only)

```python
CATALOG    = "workspace"
SCHEMA     = "analytics"
TARGET     = "TARGET"
ID_COLS    = ["client_id", "pers_id"]
MODEL_NAME = f"{CATALOG}.{SCHEMA}.diabetes_boosting_model"

SRC_TABLE   = f"{CATALOG}.{SCHEMA}.patient_raw_data"
TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.patient_train"
TEST_TABLE  = f"{CATALOG}.{SCHEMA}.patient_test"
SCORE_TABLE = f"{CATALOG}.{SCHEMA}.patient_scoring"
PRED_TABLE  = f"{CATALOG}.{SCHEMA}.diabetes_predictions"
EXPERIMENT  = f"/Shared/experiments/{SCHEMA}/diabetes"
```

### Cell 3 — Split raw data into train / test tables

```python
from medrisk_features.boosting import create_train_test_tables

split = create_train_test_tables(
    source_table=SRC_TABLE,
    train_table=TRAIN_TABLE,
    test_table=TEST_TABLE,
    target_column=TARGET,
    test_size=0.2,
    stratify=True,
    random_state=42,
    mode="overwrite",
    spark=spark,
)
print(f"Train: {split.train_count:,} rows | Test: {split.test_count:,} rows")
```

### Cell 4 — Feature engineering

```python
import pandas as pd
from medrisk_features import FeatureEngineeringPipeline
from medrisk_features.boosting import to_pandas

df = to_pandas(spark.table(TRAIN_TABLE))   # warns if > 5M rows

pipe = FeatureEngineeringPipeline(age_group_strategy="detailed", validate_schema=True)
df_features = pipe.transform(df.drop(columns=ID_COLS + [TARGET]))
df_full = pd.concat([df[ID_COLS], df_features, df[[TARGET]]], axis=1).reset_index(drop=True)

print(f"Features: {df.shape[1]} raw → {df_features.shape[1]} engineered")
```

### Cell 5 — Train and register the model

```python
from medrisk_features.boosting import TrainingConfig, train_boosting_model

config = TrainingConfig(
    target_column=TARGET,
    id_columns=ID_COLS,
    model_type="xgboost",           # swap to "catboost" or "lightgbm" freely
    model_params={"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
    experiment_name=EXPERIMENT,
    registered_model_name=MODEL_NAME,
    run_name="xgboost-v1",
    use_internal_test_split=False,   # we already have a holdout test table
)

result = train_boosting_model(df=df_full, config=config)
print(f"AUC (valid): {result.metrics['valid_roc_auc']:.4f}")
print(f"Model URI:   {result.model_uri}")
```

### Cell 6 — Evaluate on the holdout test table

```python
from medrisk_features.boosting import evaluate_registered_model_on_holdout

eval_result = evaluate_registered_model_on_holdout(
    model_uri=result.model_uri,
    holdout_table=TEST_TABLE,
    target_column=TARGET,
    spark=spark,
)
print(f"Holdout AUC: {eval_result.roc_auc:.4f} | F1: {eval_result.f1:.4f}")
```

### Cell 7 — Promote to @Champion

```python
import mlflow

mlflow.tracking.MlflowClient().set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=result.registered_model_version,
)
print(f"Version {result.registered_model_version} → @Champion")
```

### Cell 8 — Batch inference and write results back

```python
from medrisk_features.boosting import predict_with_registered_model

predict_with_registered_model(
    model_uri=f"models:/{MODEL_NAME}@Champion",
    input_table=SCORE_TABLE,
    output_table=PRED_TABLE,
    mode="overwrite",
)

display(spark.table(PRED_TABLE).limit(5))
```

---

## 9. Project Structure

```
medrisk-features/
│
├── src/medrisk_features/
│   ├── __init__.py                  ← v0.2.0, public exports
│   │
│   ├── pipeline/
│   │   └── feature_engineering_pipeline.py   ← FeatureEngineeringPipeline
│   │
│   ├── features/                    ← one module per feature domain
│   │   ├── demographics.py
│   │   ├── medical.py               ← glucose, HbA1c, BMI, BP, HOMA-IR
│   │   ├── clinical.py              ← lipid ratios, glucose interactions
│   │   ├── metabolic.py             ← dyslipidemia, cardiometabolic burden
│   │   ├── behavioral.py            ← physical activity, screen/sleep
│   │   └── lifestyle.py             ← lifestyle score
│   │
│   ├── preprocessing/
│   │   ├── categorical_cleaning.py
│   │   └── leakage.py               ← drop_leakage_columns()
│   │
│   ├── validation/
│   │   └── schema.py                ← DataSchemaValidator
│   │
│   ├── utils/
│   │   ├── constants.py             ← ADA / WHO / JNC / NCEP-ATP III thresholds
│   │   └── exceptions.py            ← MedRiskError hierarchy
│   │
│   ├── logging/
│   │   └── default_logger.py        ← Loguru singleton, get_logger()
│   │
│   ├── mlflow/
│   │   ├── pyfunc_model.py          ← MedRiskPyFuncModel
│   │   └── tracker.py               ← log_pipeline / load_pipeline / set_model_alias
│   │
│   └── boosting/
│       ├── config/
│       │   └── schemas.py           ← TrainingConfig, InferenceConfig, TrainingResult
│       ├── models/
│       │   ├── base.py              ← BaseBoostingModel (abstract)
│       │   ├── xgboost_model.py
│       │   ├── catboost_model.py
│       │   ├── lightgbm_model.py
│       │   └── factory.py           ← BoostingModelFactory.create()
│       ├── preprocessing/
│       │   └── tabular_preprocessor.py  ← TabularPreprocessor
│       ├── pyfunc/
│       │   └── boosting_pyfunc_model.py ← BoostingPyFuncModel ⭐
│       ├── training/
│       │   ├── trainer.py           ← train_boosting_model()
│       │   ├── evaluator.py         ← evaluate_splits()
│       │   └── holdout.py           ← evaluate_registered_model_on_holdout()
│       ├── inference/
│       │   └── predictor.py         ← predict_with_registered_model()
│       ├── data/
│       │   └── dataset_splitter.py  ← create_train_test_tables(), split_dataframe()
│       └── utils/
│           └── spark_utils.py       ← to_pandas(), to_spark()
│
├── notebooks/                       ← ready-to-use Databricks notebooks
│   ├── 00_environment_setup.py
│   ├── 01_create_train_test_tables.py
│   ├── 02_feature_engineering_table.py
│   ├── 03_train_register_boosting_model.py
│   ├── 04_set_model_alias.py
│   ├── 05_evaluate_registered_model.py
│   └── 06_batch_inference.py
│
├── tests/                           ← pytest test suite
│   ├── conftest.py
│   ├── test_demographics.py
│   ├── test_medical.py
│   ├── test_clinical.py
│   ├── test_metabolic.py
│   ├── test_behavioral.py
│   ├── test_lifestyle.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_schema_validation.py
│   └── test_pyfunc_model.py
│
├── .github/workflows/ci.yml        ← Lint + Format + Typecheck + Tests (3.9–3.12)
├── Makefile
└── pyproject.toml
```

---

## 10. Local Development

```bash
# Clone and install all dev dependencies
git clone https://github.com/Manda404/medrisk-features.git
cd medrisk-features
poetry install

# Run the full CI check suite locally
make check

# Individual commands
make test           # pytest + coverage (HTML report in htmlcov/)
make test-fast      # pytest without coverage (faster feedback loop)
make lint           # ruff linter
make format         # black (applies changes in place)
make format-check   # black check only (CI mode, no changes)
make typecheck      # mypy
make clean          # remove __pycache__, .coverage, htmlcov/
```

Available `make` targets:

| Target | Description |
|--------|-------------|
| `make install` | `poetry install` |
| `make test` | Full test suite with HTML coverage report |
| `make test-fast` | Tests without coverage |
| `make lint` | Ruff linter (errors only) |
| `make format` | Black auto-formatter |
| `make format-check` | Black check (no changes) |
| `make typecheck` | Mypy static analysis |
| `make check` | `lint + format-check + typecheck + test` |
| `make clean` | Remove build artifacts |

CI runs on every push to `main` and every pull request, testing Python 3.9, 3.10, 3.11, and 3.12.

---

## 11. Roadmap

### Completed ✅
- Clinical feature engineering: demographics, medical, metabolic, behavioral, lifestyle
- Schema validation with `MissingRequiredColumnError` and actionable messages
- Clinical constants module aligned with ADA 2023, WHO, JNC 7, NCEP-ATP III
- `MedRiskPyFuncModel` — feature pipeline as `mlflow.pyfunc.PythonModel`
- `BoostingPyFuncModel` — XGBoost / CatBoost / LightGBM as unified pyfunc model
- `train_boosting_model()` — one-call pipeline (split → preprocess → train → evaluate → log → register)
- `predict_with_registered_model()` — Spark + Pandas, optional Delta table output
- Holdout evaluation: `evaluate_registered_model_on_holdout()`
- Data preparation: `create_train_test_tables()`, `split_dataframe()`
- Loguru structured logging (singleton, thread-safe)
- CI/CD: ruff + black + mypy + pytest on Python 3.9–3.12
- SHAP explainability: `BoostingShapExplainer` with summary, bar, and waterfall plots

### Planned 🔜
- PyPI release — `pip install medrisk-features`
- Great Expectations schema contracts
- YAML-based training configs: `configs/training_config.yaml`
- Property-based testing with Hypothesis
- Feature drift detection and distribution monitoring
- Databricks Workflows integration (scheduled batch inference)
- Hyperparameter tuning with Optuna / Hyperopt

---

## 12. Author & License

**Rostand Surel**
📧 [s239150.eps@gmail.com](mailto:s239150.eps@gmail.com)
🔗 [github.com/Manda404](https://github.com/Manda404)

This project is released under the **MIT License** — free to use, modify, and distribute. See `LICENSE` for details.

---

Built on top of clinical guidelines (ADA, WHO, ESC, JNC, NCEP-ATP III) and open-source MLOps tooling (MLflow, Databricks Unity Catalog, scikit-learn, XGBoost, CatBoost, LightGBM).

```bibtex
@software{medrisk_features,
  author  = {Surel, Rostand},
  title   = {medrisk-features: Clinical Feature Engineering and Boosting MLOps for Healthcare ML},
  version = {0.2.0},
  year    = {2025},
  url     = {https://github.com/Manda404/medrisk-features}
}
```
