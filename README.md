# medrisk-features

![CI](https://github.com/rostandsurel/medrisk-features/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Version](https://img.shields.io/badge/version-0.2.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Databricks](https://img.shields.io/badge/Databricks-compatible-red?logo=databricks)](https://databricks.com)
[![MLflow](https://img.shields.io/badge/MLflow-pyfunc-blue?logo=mlflow)](https://mlflow.org)

**medrisk-features** is a production-ready Python package for **healthcare ML on Databricks**.

It provides two complementary layers:

| Layer | What it does |
|-------|-------------|
| **Feature Engineering** | Clinically grounded transformations — glucose, BMI, lipids, lifestyle (ADA / WHO / JNC guidelines) |
| **Boosting MLOps** | End-to-end XGBoost / CatBoost / LightGBM pipeline with `mlflow.pyfunc.PythonModel`, Unity Catalog and Databricks Workflows |

---

## 📑 Table of Contents

1. [Use Cases](#-use-cases)
2. [Architecture](#-architecture)
3. [Installation](#-installation)
4. [Feature Engineering — Quick Start](#-feature-engineering--quick-start)
5. [MLflow Integration (MedRiskPyFuncModel)](#-mlflow-integration--medriskpyfuncmodel)
6. [Boosting MLOps Layer](#-boosting-mlops-layer)
7. [🔥 Databricks Usage Guide](#-databricks-usage-guide)
   - [Step 1 — Install on Databricks](#step-1--install-on-databricks)
   - [Step 2 — Load data from Unity Catalog](#step-2--load-data-from-unity-catalog)
   - [Step 3 — Feature Engineering](#step-3--feature-engineering)
   - [Step 4 — Train a Boosting Model](#step-4--train-a-boosting-model)
   - [Step 5 — Register in Unity Catalog](#step-5--register-in-unity-catalog)
   - [Step 6 — Batch Inference](#step-6--batch-inference)
   - [Step 7 — Write results back](#step-7--write-results-back)
8. [Schema Validation](#-schema-validation)
9. [Feature Reference](#-feature-reference)
10. [Project Structure](#-project-structure)
11. [Local Development](#-local-development)
12. [Roadmap](#-roadmap)

---

## 🎯 Use Cases

- **Diabetes risk prediction** — glucose metabolism, insulin resistance, metabolic syndrome
- **Cardiometabolic risk modeling** — lipid profiles, blood pressure, BMI interactions
- **Insurance underwriting** — actuarial risk assessment with interpretable medical features
- **Clinical decision support** — production-grade features for healthcare AI systems
- **Population health analytics** — lifestyle, behavioral, and preventive risk screening

---

## 🏗️ Architecture

```
medrisk-features/
│
├── src/medrisk_features/
│   │
│   ├── pipeline/               ← FeatureEngineeringPipeline (main entry point)
│   ├── features/               ← demographics, medical, clinical, metabolic,
│   │                              behavioral, lifestyle
│   ├── preprocessing/          ← categorical cleaning, leakage removal
│   ├── validation/             ← schema validation with actionable errors
│   ├── utils/                  ← exceptions, clinical constants (ADA/WHO/JNC)
│   ├── logging/                ← Loguru-based structured logger
│   │
│   ├── mlflow/                 ← MedRiskPyFuncModel — pipeline as pyfunc model
│   │   ├── pyfunc_model.py
│   │   └── tracker.py          ← log_pipeline(), load_pipeline(), set_model_alias()
│   │
│   └── boosting/               ← Full MLOps layer for boosting models
│       ├── config/             ← TrainingConfig, InferenceConfig, TrainingResult
│       ├── models/             ← BaseBoostingModel, XGBoost, CatBoost, LightGBM, Factory
│       ├── preprocessing/      ← TabularPreprocessor (sklearn ColumnTransformer)
│       ├── pyfunc/             ← BoostingPyFuncModel (mlflow.pyfunc.PythonModel) ⭐
│       ├── training/           ← train_boosting_model(), evaluator
│       ├── inference/          ← predict_with_registered_model()
│       └── utils/              ← to_pandas(), to_spark(), write_to_delta()
│
├── notebooks/
│   ├── 01_feature_engineering_mlflow.py   ← Databricks notebook (feature pipeline)
│   └── 02_boosting_training_inference.py  ← Databricks notebook (boosting MLOps)
│
└── tests/                      ← Full unit test suite (pytest)
```

---

## 📦 Installation

### On Databricks (in a notebook cell)

```python
# Feature engineering only (no ML libs required)
%pip install git+https://github.com/Manda404/medrisk-features.git

# With MLflow integration
%pip install "git+https://github.com/Manda404/medrisk-features.git#egg=medrisk-features[mlflow]"

# Full boosting MLOps stack (recommended)
%pip install "git+https://github.com/Manda404/medrisk-features.git#egg=medrisk-features[boosting]" xgboost catboost lightgbm
```

> After `%pip install`, always call `dbutils.library.restartPython()` before importing.

### Standard pip

```bash
# Core package (feature engineering only)
pip install git+https://github.com/Manda404/medrisk-features.git

# With optional extras
pip install "medrisk-features[mlflow]"      # + MLflow
pip install "medrisk-features[xgboost]"     # + XGBoost
pip install "medrisk-features[catboost]"    # + CatBoost
pip install "medrisk-features[lightgbm]"    # + LightGBM
pip install "medrisk-features[boosting]"    # + all boosting libs + sklearn + mlflow
pip install "medrisk-features[all]"         # everything
```

### With Poetry

```bash
poetry add git+https://github.com/Manda404/medrisk-features.git
```

### Development

```bash
git clone https://github.com/Manda404/medrisk-features.git
cd medrisk-features
poetry install
make test        # run tests
make lint        # ruff + black check
make typecheck   # mypy
```

---

## 🚀 Feature Engineering — Quick Start

### Basic usage

```python
import pandas as pd
from medrisk_features import FeatureEngineeringPipeline

df = pd.read_csv("patient_data.csv")

pipeline = FeatureEngineeringPipeline(
    age_group_strategy="detailed",  # "detailed" | "coarse"
    validate_schema=True,
)

df_enriched = pipeline.transform(df)
# Raw columns: 19 → Enriched columns: 40+
print(df_enriched.columns.tolist())
```

### Integration with scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from medrisk_features import FeatureEngineeringPipeline

ml_pipeline = Pipeline([
    ("features",    FeatureEngineeringPipeline()),
    ("scaler",      StandardScaler()),
    ("classifier",  XGBClassifier()),
])
ml_pipeline.fit(X_train, y_train)
```

### Error handling

```python
from medrisk_features import FeatureEngineeringPipeline, MissingRequiredColumnError

try:
    df_enriched = pipeline.transform(df)
except MissingRequiredColumnError as e:
    print(f"Missing columns: {e.missing_columns}")
    # Output: Missing columns: ['glucose_fasting', 'bmi']
```

---

## 🔬 MLflow Integration — MedRiskPyFuncModel

The **feature engineering pipeline** itself can be logged to MLflow as a `PythonModel`, versioned in the Unity Catalog Model Registry, and reloaded for inference anywhere MLflow is available.

```python
from medrisk_features import FeatureEngineeringPipeline
from medrisk_features.mlflow import log_pipeline, load_pipeline, set_model_alias

# 1. Build and run pipeline
pipeline = FeatureEngineeringPipeline(validate_schema=True)
df_enriched = pipeline.transform(df)

# 2. Log to MLflow as a pyfunc model
result = log_pipeline(
    pipeline=pipeline,
    experiment_name="/Shared/experiments/medrisk",
    df_sample=df.head(20),
    registered_model_name="workspace.my_schema.medrisk_pipeline",
    tags={"team": "data-science", "domain": "healthcare"},
)
print(result.model_uri)
# → runs:/abc123/medrisk_pipeline

# 3. Assign @Champion alias (Unity Catalog)
set_model_alias(
    registered_model_name="workspace.my_schema.medrisk_pipeline",
    version=result.registered_model_version,
    alias="Champion",
)

# 4. Load and run inference
loaded = load_pipeline("models:/workspace.my_schema.medrisk_pipeline@Champion")
df_out = loaded.predict(df_raw)
```

---

## ⚡ Boosting MLOps Layer

The `medrisk_features.boosting` subpackage provides a complete MLOps pipeline for training, logging, registering and serving XGBoost / CatBoost / LightGBM models.

### The core class: `BoostingPyFuncModel`

```python
class BoostingPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Loads: model + preprocessor + config + feature_names

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        # 1. Validate input
        # 2. Select & order feature columns
        # 3. Apply TabularPreprocessor (imputation + encoding)
        # 4. Predict probability (XGBoost / CatBoost / LightGBM)
        # 5. Apply priority rules (HIGH / MEDIUM / LOW)
        # 6. Return clean DataFrame
```

**Output format** (configurable, default):

| Column | Description |
|--------|-------------|
| `client_id`, `pers_id` | Identifier columns (from `id_columns`) |
| `probability` | Positive-class probability (0–1) |
| `priority` | `HIGH` ≥ 0.8 · `MEDIUM` ≥ 0.5 · `LOW` otherwise |
| `prediction_date` | UTC timestamp |
| `model_version` | Registered model version |

### Training

```python
from medrisk_features.boosting import TrainingConfig, train_boosting_model

config = TrainingConfig(
    target_column="TARGET",
    id_columns=["client_id", "pers_id"],
    model_type="xgboost",        # "xgboost" | "catboost" | "lightgbm"
    model_params={
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
    },
    experiment_name="/Shared/experiments/boosting",
    registered_model_name="workspace.my_schema.boosting_model",
    priority_high_threshold=0.8,
    priority_medium_threshold=0.5,
)

result = train_boosting_model(df=df_train, config=config)

print(result.metrics["test_roc_auc"])   # e.g. 0.87
print(result.model_uri)                  # runs:/abc.../boosting_model
```

### Inference

```python
from medrisk_features.boosting import predict_with_registered_model

predictions = predict_with_registered_model(
    model_uri="models:/workspace.my_schema.boosting_model@Champion",
    input_df=df_new,       # Pandas or Spark DataFrame
)
```

---

## 🔥 Databricks Usage Guide

This section walks through the complete end-to-end workflow on Databricks, from data ingestion to model serving.

---

### Step 1 — Install on Databricks

In the **first cell** of your notebook:

```python
# Install the package with the full boosting stack
%pip install \
  "git+https://github.com/Manda404/medrisk-features.git#egg=medrisk-features[boosting]" \
  xgboost catboost lightgbm

# Always restart Python after %pip install
dbutils.library.restartPython()
```

> **Tip:** For a persistent cluster, add the package to your cluster's **Init script** or **Libraries** tab to avoid reinstalling on every run.

---

### Step 2 — Load data from Unity Catalog

```python
# Load a Delta table from Unity Catalog as a Spark DataFrame
df_spark = spark.table("workspace.my_schema.patient_training_data")

# Convert to Pandas for feature engineering and training
# (medrisk-features handles the warning if the dataset is very large)
from medrisk_features.boosting import to_pandas

df = to_pandas(df_spark)  # warns if > 5M rows
print(f"Loaded {len(df):,} rows — {df.shape[1]} columns")
```

---

### Step 3 — Feature Engineering

```python
from medrisk_features import FeatureEngineeringPipeline, MissingRequiredColumnError

ID_COLUMNS    = ["client_id", "pers_id"]
TARGET_COLUMN = "TARGET"

# Run the clinical feature engineering pipeline
pipeline = FeatureEngineeringPipeline(
    age_group_strategy="detailed",
    validate_schema=True,
)

df_features = df.drop(columns=ID_COLUMNS + [TARGET_COLUMN])

try:
    df_engineered = pipeline.transform(df_features)
except MissingRequiredColumnError as e:
    print(f"Missing: {e.missing_columns}")
    raise

# Reassemble: identifiers + features + target
import pandas as pd
df_full = pd.concat([
    df[ID_COLUMNS].reset_index(drop=True),
    df_engineered.reset_index(drop=True),
    df[[TARGET_COLUMN]].reset_index(drop=True),
], axis=1)

print(f"Enriched: {df_features.shape[1]} → {df_engineered.shape[1]} feature columns")
display(df_full.head(3))
```

---

### Step 4 — Train a Boosting Model

```python
from medrisk_features.boosting import TrainingConfig, train_boosting_model

config = TrainingConfig(
    # ── Data ─────────────────────────────────────────────────────────────
    target_column=TARGET_COLUMN,
    id_columns=ID_COLUMNS,

    # ── Model ────────────────────────────────────────────────────────────
    model_type="xgboost",   # swap to "catboost" or "lightgbm" anytime
    model_params={
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    # ── Split ────────────────────────────────────────────────────────────
    test_size=0.2,
    validation_size=0.2,
    random_state=42,

    # ── MLflow ───────────────────────────────────────────────────────────
    experiment_name="/Shared/experiments/my_schema/diabetes_risk",
    registered_model_name="workspace.my_schema.diabetes_boosting_model",
    run_name="xgboost-v1",

    # ── Output priority thresholds ────────────────────────────────────────
    priority_high_threshold=0.8,
    priority_medium_threshold=0.5,
)

result = train_boosting_model(df=df_full, config=config)

print(f"✅ Training complete")
print(f"   Run ID    : {result.run_id}")
print(f"   Model URI : {result.model_uri}")
print(f"   AUC test  : {result.metrics.get('test_roc_auc', 'N/A'):.4f}")
print(f"   F1 test   : {result.metrics.get('test_f1', 'N/A'):.4f}")
```

> `train_boosting_model()` automatically handles:  
> stratified split · preprocessing · training · evaluation on 3 splits · MLflow logging of params, metrics, artifacts, signature · pyfunc model logging · Model Registry registration.

---

### Step 5 — Register in Unity Catalog

```python
import mlflow

# Assign the @Champion alias to promote the model to production
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(
    name="workspace.my_schema.diabetes_boosting_model",
    alias="Champion",
    version=result.registered_model_version,
)
print(f"✅ Version {result.registered_model_version} promoted to @Champion")
```

You can also load any specific version by alias or stage:

```python
# By Unity Catalog alias (recommended)
model_uri = "models:/workspace.my_schema.diabetes_boosting_model@Champion"

# By version number
model_uri = "models:/workspace.my_schema.diabetes_boosting_model/3"

# By run URI (no registry)
model_uri = result.model_uri
```

---

### Step 6 — Batch Inference

The `predict_with_registered_model()` function handles everything: loading the model, converting Spark DataFrames if needed, applying preprocessing, and returning clean predictions.

```python
from medrisk_features.boosting import predict_with_registered_model

# Option A — from a Pandas DataFrame
predictions = predict_with_registered_model(
    model_uri="models:/workspace.my_schema.diabetes_boosting_model@Champion",
    input_df=df_scoring,   # Pandas or Spark — both accepted
)

# Option B — directly from a Unity Catalog table
predictions = predict_with_registered_model(
    model_uri="models:/workspace.my_schema.diabetes_boosting_model@Champion",
    input_table="workspace.my_schema.patient_scoring_data",
)

display(predictions)
```

**Output example:**

| client_id | pers_id | probability | priority | prediction_date | model_version |
|-----------|---------|-------------|----------|-----------------|---------------|
| C00001 | P00001 | 0.923 | HIGH | 2025-06-15 09:42:11 | 3 |
| C00002 | P00002 | 0.641 | MEDIUM | 2025-06-15 09:42:11 | 3 |
| C00003 | P00003 | 0.187 | LOW | 2025-06-15 09:42:11 | 3 |

---

### Step 7 — Write results back

```python
# Write predictions to a Unity Catalog Delta table
predict_with_registered_model(
    model_uri="models:/workspace.my_schema.diabetes_boosting_model@Champion",
    input_table="workspace.my_schema.patient_scoring_data",
    output_table="workspace.my_schema.diabetes_predictions",
    mode="overwrite",   # or "append"
)

# Verify
display(spark.table("workspace.my_schema.diabetes_predictions").limit(10))
```

---

### Complete end-to-end summary (copy-paste template)

```python
# ── Cell 1: Install ───────────────────────────────────────────────────────────
%pip install "git+https://github.com/Manda404/medrisk-features.git#egg=medrisk-features[boosting]" xgboost catboost lightgbm
dbutils.library.restartPython()

# ── Cell 2: Imports & config ──────────────────────────────────────────────────
import pandas as pd
from medrisk_features import FeatureEngineeringPipeline
from medrisk_features.boosting import (
    TrainingConfig, train_boosting_model, predict_with_registered_model, to_pandas
)

CATALOG       = "workspace"
SCHEMA        = "my_schema"
TARGET        = "TARGET"
ID_COLS       = ["client_id", "pers_id"]
MODEL_NAME    = f"{CATALOG}.{SCHEMA}.diabetes_boosting_model"

# ── Cell 3: Load data ─────────────────────────────────────────────────────────
df = to_pandas(spark.table(f"{CATALOG}.{SCHEMA}.training_data"))

# ── Cell 4: Feature engineering ───────────────────────────────────────────────
pipe = FeatureEngineeringPipeline(validate_schema=True)
df_eng = pipe.transform(df.drop(columns=ID_COLS + [TARGET]))
df_full = pd.concat([df[ID_COLS], df_eng, df[[TARGET]]], axis=1)

# ── Cell 5: Train ─────────────────────────────────────────────────────────────
config = TrainingConfig(
    target_column=TARGET, id_columns=ID_COLS,
    model_type="xgboost",
    experiment_name=f"/Shared/experiments/{SCHEMA}/diabetes",
    registered_model_name=MODEL_NAME,
)
result = train_boosting_model(df=df_full, config=config)
print(f"AUC: {result.metrics['test_roc_auc']:.4f}")

# ── Cell 6: Promote to Champion ───────────────────────────────────────────────
import mlflow
mlflow.tracking.MlflowClient().set_registered_model_alias(
    name=MODEL_NAME, alias="Champion",
    version=result.registered_model_version,
)

# ── Cell 7: Inference ─────────────────────────────────────────────────────────
predictions = predict_with_registered_model(
    model_uri=f"models:/{MODEL_NAME}@Champion",
    input_table=f"{CATALOG}.{SCHEMA}.scoring_data",
    output_table=f"{CATALOG}.{SCHEMA}.predictions",
)
display(predictions.head(10))
```

---

## 🔐 Schema Validation

The pipeline enforces a **minimum required schema** before any transformation.

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `Age` | numeric | Patient age in years |
| `glucose_fasting` | numeric | Fasting blood glucose (mg/dL) |
| `bmi` | numeric | Body mass index (kg/m²) |

### Optional columns (unlock additional features)

| Column(s) | Features unlocked |
|-----------|------------------|
| `hba1c` | `hba1c_category` |
| `insulin_level` | `homa_ir`, `insulin_resistance_flag` |
| `systolic_bp`, `diastolic_bp` | `bp_category`, `blood_pressure_ratio` |
| `triglycerides`, `hdl_cholesterol` | `dyslipidemia_flag`, `metabolic_syndrome_flag` |
| `hdl_cholesterol`, `ldl_cholesterol` | `lipid_ratio_hdl_ldl` |
| `physical_activity_minutes_per_week` | `physical_activity_adequate` |
| `screen_time_hours_per_day`, `sleep_hours_per_day` | `screen_sleep_imbalance`, `sleep_efficiency` |
| `diet_score`, `smoking_status`, `alcohol_consumption_per_week` | `lifestyle_score` |

```python
from medrisk_features import MissingRequiredColumnError

try:
    df_out = pipeline.transform(df)
except MissingRequiredColumnError as e:
    print(f"Schema error: {e}")
    print(f"Missing: {e.missing_columns}")
```

---

## 🧬 Feature Reference

All feature names match the documented identifiers and align with clinical guidelines.

### Demographics
| Feature | Description | Guideline |
|---------|-------------|-----------|
| `age_group` | Age band (detailed or coarse) | — |
| `age_squared` | Non-linear age effect | — |
| `socioeconomic_vulnerability_flag` | Low income + low education | — |

### Medical
| Feature | Description | Guideline |
|---------|-------------|-----------|
| `glucose_category` | Normal / Pre-Diabetes / Diabetes | ADA 2023 |
| `hba1c_category` | Glycemic control stratification | ADA 2023 |
| `bmi_category` | Underweight / Normal / Overweight / Obese | WHO |
| `bp_category` | Normal / Pre-Hypertension / Hypertension | JNC 7 |
| `homa_ir` | Insulin resistance index | — |
| `insulin_resistance_flag` | HOMA-IR > 2.5 | — |
| `metabolic_syndrome_flag` | ≥ 3 ATP-III criteria | NCEP-ATP III |

### Clinical Interactions
| Feature | Description |
|---------|-------------|
| `lipid_ratio_hdl_ldl` | HDL / LDL atherogenic ratio |
| `cholesterol_hdl_ratio` | Total cholesterol / HDL ratio |
| `bmi_glucose_interaction` | BMI × fasting glucose synergy |
| `glucose_variability` | Postprandial − fasting glucose excursion |

### Metabolic
| Feature | Description |
|---------|-------------|
| `glycemic_load` | Glucose × BMI burden proxy |
| `dyslipidemia_flag` | Triglycerides ≥ 150 or HDL < 40 |
| `cardiometabolic_burden` | 0–5 composite risk score |
| `blood_pressure_ratio` | Systolic / diastolic ratio |

### Behavioral
| Feature | Description |
|---------|-------------|
| `physical_activity_adequate` | Activity / WHO 150 min/week ratio |
| `screen_sleep_imbalance` | Screen time / sleep hours ratio |
| `sedentary_risk` | High screen + low activity flag |

### Lifestyle
| Feature | Description |
|---------|-------------|
| `lifestyle_score` | Global health score 0–10 |
| `sleep_efficiency` | Sleep hours / (screen time + 1) |

---

## 🏗️ Project Structure

```
medrisk-features/
│
├── src/medrisk_features/
│   ├── __init__.py                 ← v0.2.0, public exports
│   ├── pipeline/
│   │   └── feature_engineering_pipeline.py
│   ├── features/
│   │   ├── demographics.py
│   │   ├── medical.py
│   │   ├── clinical.py
│   │   ├── metabolic.py
│   │   ├── behavioral.py
│   │   └── lifestyle.py
│   ├── preprocessing/
│   │   ├── categorical_cleaning.py
│   │   └── leakage.py
│   ├── validation/
│   │   └── schema.py
│   ├── utils/
│   │   ├── exceptions.py           ← MedRiskError, MissingRequiredColumnError…
│   │   └── constants.py            ← ADA/WHO/JNC clinical thresholds
│   ├── logging/
│   │   └── default_logger.py
│   │
│   ├── mlflow/
│   │   ├── pyfunc_model.py         ← MedRiskPyFuncModel
│   │   └── tracker.py              ← log_pipeline / load_pipeline / set_model_alias
│   │
│   └── boosting/
│       ├── config/
│       │   └── schemas.py          ← TrainingConfig, InferenceConfig, TrainingResult
│       ├── models/
│       │   ├── base.py             ← BaseBoostingModel (abstract)
│       │   ├── xgboost_model.py
│       │   ├── catboost_model.py
│       │   ├── lightgbm_model.py
│       │   └── factory.py          ← BoostingModelFactory
│       ├── preprocessing/
│       │   └── tabular_preprocessor.py
│       ├── pyfunc/
│       │   └── boosting_pyfunc_model.py  ← BoostingPyFuncModel ⭐
│       ├── training/
│       │   ├── trainer.py          ← train_boosting_model()
│       │   └── evaluator.py
│       ├── inference/
│       │   └── predictor.py        ← predict_with_registered_model()
│       └── utils/
│           └── spark_utils.py      ← to_pandas / to_spark / write_to_delta
│
├── notebooks/
│   ├── 01_feature_engineering_mlflow.py
│   └── 02_boosting_training_inference.py
│
├── tests/
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
├── .github/workflows/ci.yml       ← Lint + Format + Typecheck + Tests (Python 3.9/3.10/3.11)
├── Makefile                        ← make test / lint / format / typecheck / check
├── pyproject.toml
└── README.md
```

---

## 🛠️ Local Development

```bash
# Clone and install all dev dependencies
git clone https://github.com/Manda404/medrisk-features.git
cd medrisk-features
poetry install

# Run all checks (equivalent of CI)
make check

# Individual commands
make test       # pytest with coverage
make lint       # ruff linter
make format     # black formatter
make typecheck  # mypy
make clean      # remove __pycache__, .coverage, htmlcov
```

### Available `make` targets

| Target | Description |
|--------|-------------|
| `make install` | Install all dependencies |
| `make test` | Run tests with HTML coverage report |
| `make test-fast` | Run tests without coverage (faster) |
| `make lint` | ruff linter check |
| `make format` | black formatter (applies changes) |
| `make format-check` | black check only (for CI) |
| `make typecheck` | mypy type checking |
| `make check` | lint + format-check + typecheck + test |
| `make clean` | Remove build artifacts |

---

## 🛣️ Roadmap

### Done ✅
- [x] Clinical feature engineering (demographics, medical, metabolic, behavioral, lifestyle)
- [x] Schema validation with `MissingRequiredColumnError`
- [x] Clinical constants module (ADA / WHO / JNC / NCEP-ATP III)
- [x] `MedRiskPyFuncModel` — feature pipeline as `mlflow.pyfunc.PythonModel`
- [x] `BoostingPyFuncModel` — XGBoost / CatBoost / LightGBM with full MLflow integration
- [x] `train_boosting_model()` — one-call training with stratified split, preprocessing, MLflow
- [x] `predict_with_registered_model()` — Spark + Pandas, Unity Catalog output
- [x] `spark_utils` — safe Spark ↔ Pandas conversion with large-dataset warnings
- [x] CI/CD: lint (ruff) + format (black) + typecheck (mypy) + tests (3.9/3.10/3.11)
- [x] Makefile + full test suite (10 test files)

### Planned 🔜
- [ ] 📦 PyPI release — `pip install medrisk-features`
- [ ] 🔍 SHAP explainability helpers (`shap_summary_plot`, feature contribution DataFrame)
- [ ] 📐 Great Expectations schema contracts
- [ ] ⚙️ YAML-based config files (`configs/training_config.yaml`)
- [ ] 🧪 Property-based testing with Hypothesis
- [ ] 📊 Drift detection (feature distribution monitoring)
- [ ] 🔄 Databricks Workflows integration (scheduled batch inference)
- [ ] 🌐 Multi-language support (French medical terminology)
- [ ] 🚀 Hyperparameter tuning with Optuna / Hyperopt

---

## 🧠 Design Philosophy

1. **Explainability First** — every feature has clear clinical meaning and a documented source guideline
2. **No Silent Failures** — explicit `MissingRequiredColumnError` with actionable messages
3. **Optional Dependencies** — core package installs without ML libs; extras are opt-in
4. **Databricks-Native** — Spark DataFrames accepted everywhere, Unity Catalog first-class
5. **pyfunc First** — model output is always a `mlflow.pyfunc.PythonModel`, never a raw sklearn pipeline
6. **Separation of Concerns** — preprocessing, training, inference, and MLflow logging are fully decoupled

---

## 👤 Author

**Rostand Surel**
📧 [rostandsurel@yahoo.com](mailto:rostandsurel@yahoo.com)
🔗 [GitHub](https://github.com/Manda404)

---

## 📄 License

MIT License — free to use, modify and distribute. See `LICENSE` for details.

---

## 🙏 Acknowledgments

Built on top of:
- Clinical guidelines: ADA, WHO, ESC, JNC, NCEP-ATP III
- MLOps patterns: MLflow, Databricks Unity Catalog
- ML engineering: scikit-learn, XGBoost, CatBoost, LightGBM

---

## 📚 Citation

```bibtex
@software{medrisk_features,
  author  = {Surel, Rostand},
  title   = {medrisk-features: Clinical Feature Engineering and Boosting MLOps for Healthcare ML},
  version = {0.2.0},
  year    = {2025},
  url     = {https://github.com/Manda404/medrisk-features}
}
```

---

**Questions?** Open an issue or contact [rostandsurel@yahoo.com](mailto:rostandsurel@yahoo.com)
