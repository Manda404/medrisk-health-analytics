# MedRisk Health Analytics

![CI](https://github.com/Manda404/medrisk-health-analytics/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
![Version](https://img.shields.io/badge/version-0.4.1-orange)

Package Python pour préparer des données de patients, créer des variables métier,
entraîner des modèles de classification et exécuter un pipeline MLOps sur
Databricks.

> Projet d’analyse et d’apprentissage. Il ne remplace pas un avis médical.

## Installation

```bash
poetry install
```

Ou depuis le dépôt GitHub :

```bash
pip install "git+https://github.com/Manda404/medrisk-health-analytics.git"
```

## Utilisation rapide

```python
from medrisk_health_analytics import (
    DatasetAnalyzer,
    DatasetLoader,
    DatasetPreprocessor,
    FeatureEngineeringPipeline,
)

# 1. Charger les données
data = DatasetLoader().load("data/patient.csv")

# 2. Examiner leur qualité
analysis = DatasetAnalyzer(
    target_column="diagnosed_diabetes"
).analyze(data)

print(f"Nombre de patients : {analysis.rows}")
print(f"Nombre de colonnes : {analysis.columns}")
print(f"Nombre de doublons : {analysis.duplicate_rows}")
print(f"Valeurs manquantes : {analysis.missing_values}")
print(f"Distribution cible : {analysis.target_distribution}")

# 3. Nettoyer les données
clean_data = DatasetPreprocessor().transform(data)

# 4. Construire les variables métier
features = FeatureEngineeringPipeline().transform(
    clean_data,
    target_column="diagnosed_diabetes",
)

print(features.head())
```

## Charger différents formats

```python
from medrisk_health_analytics import DatasetLoader

csv_data = DatasetLoader().load("patients.csv")
parquet_data = DatasetLoader().load("patients.parquet")
json_data = DatasetLoader().load("patients.json")
```

Depuis une table Unity Catalog dans un notebook Databricks :

```python
from medrisk_health_analytics import DatasetLoader

data = DatasetLoader(spark=spark).load(
    "workspace.mlops_dev.patient_raw_data"
)
```

## Valider les données

```python
from medrisk_health_analytics import DataQualityValidator

report = DataQualityValidator(
    required_columns=(
        "age",
        "glucose_fasting",
        "bmi",
        "diagnosed_diabetes",
    ),
    target_column="diagnosed_diabetes",
    max_missing_ratio=0.05,
).validate(data)

print(report.passed)
print(report.violations)

# Arrête le traitement si le contrat n'est pas respecté.
report.raise_for_failure()
```

## Entraîner et enregistrer un modèle

```python
from medrisk_health_analytics import ModelTrainer

trainer = ModelTrainer(
    target_column="diagnosed_diabetes",
    model_type="xgboost",
    experiment_name="/Shared/medrisk-health-analytics",
    registered_model_name="workspace.mlops_dev.boosting_risk_model",
    run_name="medrisk-xgboost",
)

result = trainer.fit(features)

print(result.run_id)
print(result.model_uri)
print(result.registered_model_version)
print(result.metrics)
```

Modèles disponibles :

```text
xgboost
catboost
lightgbm
logistic_regression
random_forest
gradient_boosting
```

## Utiliser directement un modèle

```python
from medrisk_health_analytics import RandomForestModel

model = RandomForestModel(
    params={"n_estimators": 300, "random_state": 42}
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

Les wrappers disponibles partagent la même interface :

```python
from medrisk_health_analytics import (
    CatBoostModel,
    GradientBoostingModel,
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
```

## Évaluer un modèle

```python
from medrisk_health_analytics import ModelEvaluator

evaluator = ModelEvaluator(
    model,
    model_name="diabetes-random-forest",
    threshold=0.5,
    quality_gates={
        "roc_auc": 0.75,
        "recall": 0.70,
        "mcc": 0.40,
    },
)

evaluation = evaluator.evaluate(X_test, y_test)

print(evaluation.metrics)
print(evaluation.passed)
print(evaluation.confusion_matrix)
print(evaluation.classification_report)
```

Le rapport inclut notamment ROC AUC, Average Precision, recall, spécificité,
MCC, F1, balanced accuracy, valeur prédictive négative et score de Brier.

## Faire des prédictions avec MLflow

```python
from medrisk_health_analytics import ModelPredictor

predictor = ModelPredictor(
    "models:/workspace.mlops_dev.boosting_risk_model@Champion"
)

predictions = predictor.predict(new_patient_features)
print(predictions.head())
```

## Mesurer la dérive des données

```python
from medrisk_health_analytics import DataDriftMonitor

monitor = DataDriftMonitor(psi_threshold=0.20)
drift = monitor.compare(
    reference=training_features,
    current=new_patient_features,
)

print(drift.passed)
print(drift.drifted_features)
print(drift.metrics)
```

## Exécuter le pipeline Databricks

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run medrisk_classification_pipeline -t dev
```

Le workflow exécute successivement :

```text
préparation → entraînement → validation → promotion → prédiction → surveillance
```

Les tables de développement sont enregistrées dans :

```text
workspace.mlops_dev
```

## Vérifier le projet

```bash
make check
```

Cette commande exécute le lint, le formatage, le contrôle des types, les tests et
la construction du package.

## Documentation détaillée

- [Architecture MLOps Databricks](docs/MLOPS_DATABRICKS.md)
- [Variables métier](docs/FEATURE_ENGINEERING.md)
- [Guide des notebooks](docs/NOTEBOOKS.md)
- [Audit technique](docs/AUDIT.md)
- [Contribution](docs/CONTRIBUTING.md)
- [Sécurité](docs/SECURITY.md)

## Licence

MIT — consultez [LICENSE](LICENSE).
