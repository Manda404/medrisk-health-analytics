# MedRisk Health Analytics

![CI](https://github.com/Manda404/medrisk-health-analytics/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
![Version](https://img.shields.io/badge/version-0.4.1-orange)

MedRisk Health Analytics est un package Python qui prépare des données de
patients, construit des variables métier et entraîne un modèle de classification
du risque de diabète. Le projet peut être exécuté localement ou dans un workflow
MLOps Databricks.

> Ce projet sert à l’analyse et à l’apprentissage. Il ne remplace pas un avis
> médical et ne doit pas être utilisé seul pour établir un diagnostic.

## Problématique

Le fichier `data/patient.csv` contient des mesures cliniques, des informations
démographiques, des antécédents et des habitudes de vie. La colonne
`diagnosed_diabetes` indique la classe que le modèle doit apprendre à prédire.

Utiliser directement ce fichier pose plusieurs problèmes :

- la qualité des données doit être vérifiée avant l’entraînement ;
- une mesure isolée représente parfois mal le risque métabolique ;
- les transformations doivent rester identiques entre entraînement et prédiction ;
- `diabetes_risk_score` et `diabetes_stage` révèlent directement le diagnostic et
  provoqueraient une fuite de cible ;
- un modèle ne doit être publié qu’après une évaluation sur des données séparées ;
- les nouvelles données doivent être surveillées pour détecter une dérive.

Le package rassemble ces règles dans des classes réutilisables. Les notebooks
Databricks servent uniquement à orchestrer le même code sur les tables Unity
Catalog.

## Installation locale

```bash
poetry install
```

## Charger, analyser et préparer les données

Cet exemple utilise le fichier réellement présent dans le dépôt et les classes
publiques du package :

```python
from medrisk_health_analytics import (
    DatasetAnalyzer,
    DatasetLoader,
    DatasetPreprocessor,
    FeatureEngineeringPipeline,
)

# Charger le fichier data/patient.csv dans un DataFrame Pandas.
data = DatasetLoader().load("data/patient.csv")

# Examiner sa taille, ses doublons, ses valeurs manquantes et sa cible.
analysis = DatasetAnalyzer(
    target_column="diagnosed_diabetes"
).analyze(data)

print(f"Nombre de patients : {analysis.rows}")
print(f"Nombre de colonnes : {analysis.columns}")
print(f"Nombre de doublons : {analysis.duplicate_rows}")
print(f"Valeurs manquantes : {analysis.missing_values}")
print(f"Distribution cible : {analysis.target_distribution}")

# Nettoyer les catégories, les valeurs infinies et les doublons.
clean_data = DatasetPreprocessor().transform(data)

# Créer les variables métier tout en conservant la cible.
features = FeatureEngineeringPipeline().transform(
    clean_data,
    target_column="diagnosed_diabetes",
)

print(features.head())
```

`FeatureEngineeringPipeline` crée notamment la pression artérielle moyenne, la
pression pulsée, le ratio triglycérides/HDL, l’indice TyG, l’indice TyG-IMC et des
indicateurs liés aux antécédents, au sommeil et à l’activité physique.

Les colonnes de fuite `diabetes_risk_score` et `diabetes_stage` sont retirées des
variables utilisables par le modèle.

## Vérifier le contrat des données

L’exemple suivant réutilise la variable `data` créée précédemment :

```python
from medrisk_health_analytics import DataQualityValidator

quality = DataQualityValidator(
    required_columns=(
        "age",
        "glucose_fasting",
        "bmi",
        "diagnosed_diabetes",
    ),
    target_column="diagnosed_diabetes",
    max_missing_ratio=0.05,
).validate(data)

print(quality.passed)
print(quality.violations)

# Déclenche une erreur explicite si le contrat n'est pas respecté.
quality.raise_for_failure()
```

## Entraîner un modèle

L’exemple réutilise `features`, construit dans le premier exemple :

```python
from medrisk_health_analytics import ModelTrainer

trainer = ModelTrainer(
    target_column="diagnosed_diabetes",
    model_type="xgboost",
    experiment_name="/Shared/medrisk-health-analytics",
    run_name="medrisk-xgboost",
)

result = trainer.fit(features)

print(f"Run MLflow : {result.run_id}")
print(f"Modèle : {result.model_uri}")
print(result.metrics)
```

Les valeurs acceptées pour `model_type` sont celles implémentées dans le package :

```text
xgboost
catboost
lightgbm
logistic_regression
random_forest
gradient_boosting
```

L’entraînement sépare automatiquement une partie des données pour la validation.
Il enregistre dans MLflow la configuration, les métriques, les variables, le
préprocesseur, le modèle, sa signature et un exemple d’entrée.

## Prédire avec le modèle entraîné

Cet exemple utilise directement `result.model_uri`. Il retire la cible de cinq
lignes du DataFrame `features` avant la prédiction :

```python
from medrisk_health_analytics import ModelPredictor

patients_to_score = features.drop(
    columns=["diagnosed_diabetes"]
).head(5)

predictor = ModelPredictor(result.model_uri)
predictions = predictor.predict(patients_to_score)

print(predictions)
```

La sortie contient la probabilité estimée, une priorité et les informations de
version du modèle.

## Exécuter le projet sur Databricks

La source utilisée par le workflow se trouve dans :

```text
/Volumes/workspace/mlops_dev/medrisk_data/patient.csv
```

Les tables et le modèle sont enregistrés dans `workspace.mlops_dev`.

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run medrisk_classification_pipeline -t dev
```

Le job exécute quatre tâches dans cet ordre :

```text
prepare_data
    → train_validate_promote
    → batch_inference
    → monitor_data_drift
```

Le candidat est évalué sur une table holdout séparée. Il reçoit l’alias
`Champion` uniquement s’il respecte les seuils configurés. La dernière tâche
compare les données de scoring à la référence d’entraînement et enregistre les
mesures de dérive dans une table Delta.

## Vérifier le projet

```bash
make check
```

Cette commande exécute les tests, le lint, le formatage, le contrôle des types et
la construction du package.

## Licence

MIT — consultez [LICENSE](LICENSE).
