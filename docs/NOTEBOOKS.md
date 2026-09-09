# Guide des notebooks Databricks

La logique métier appartient au package Python. Les trois notebooks servent
uniquement à orchestrer les opérations Databricks.

| Ordre | Notebook | Responsabilité |
|---:|---|---|
| 1 | `01_prepare_data.py` | Ingestion, nettoyage, feature engineering et séparation train/holdout |
| 2 | `02_train_validate_promote.py` | Entraînement, enregistrement MLflow, validation et promotion |
| 3 | `03_batch_inference.py` | Prédictions avec le modèle portant l’alias configuré |
| 4 | `04_monitor_data_drift.py` | PSI par variable et historisation du drift dans Delta |

La première cellule de `01_prepare_data.py` vérifie que la wheel est installée
et affiche les versions du package et de Python. Une installation invalide
arrête donc le job avant toute écriture de données.

## Pourquoi quatre notebooks ?

- La préparation peut être relancée sans réentraîner immédiatement le modèle.
- Une erreur d’entraînement empêche la promotion et l’inférence.
- L’inférence reste une étape clairement observable dans le job.
- La surveillance peut être relancée sans produire de nouvelles prédictions.
- Aucun grand notebook ne mélange toutes les responsabilités.

La validation utilise directement `training_result.model_uri`, puis la promotion
utilise `training_result.registered_model_version`. Comme ces opérations sont
dans le même notebook, `dbutils.jobs.taskValues` n’est plus nécessaire.

## Classes principales

```python
loader = DatasetLoader(spark=spark)
preprocessor = DatasetPreprocessor()
pipeline = FeatureEngineeringPipeline()
trainer = ModelTrainer(...)
predictor = ModelPredictor(model_uri)
```

Les conversions `toPandas()` chargent les données dans la mémoire du driver.
Pour de grands volumes, il faudra ajouter un contrôle de taille ou adopter un
entraînement distribué.
