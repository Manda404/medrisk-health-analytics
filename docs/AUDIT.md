# Audit technique de MedRisk Health Analytics

Date : 8 septembre 2026  
Version auditée : 0.2.0

## Conclusion

Le projet possédait une base fonctionnelle, mais son dépôt n'était pas fiable pour une livraison : deux packages dupliqués coexistaient, un module Python était masqué par le `.gitignore`, le contrôle statique échouait et une branche d'exécution du pipeline médical référençait une constante non importée.

Les problèmes confirmés ont été corrigés. Le package canonique est désormais `medrisk_health_analytics`.

## Corrections réalisées

### Structure et packaging

- Suppression de l'ancienne copie `src/medrisk_features` afin d'éviter deux implémentations divergentes.
- Correction de la règle `.gitignore` : `data` est devenu `/data/`. Le module `src/medrisk_health_analytics/boosting/data` peut maintenant être versionné et inclus dans le paquet.
- Correction de l'arborescence présentée dans le README.
- Correction des commandes du Makefile pour qu'elles contrôlent le package réellement publié.
- Régénération de `poetry.lock` pour le synchroniser avec `pyproject.toml`.
- Ajout de `.DS_Store` au `.gitignore`.

### Fiabilité fonctionnelle

- Import de `BP_DIASTOLIC_PREHYPERTENSION_MAX`, qui supprimait un `NameError` lors du calcul de certaines catégories de pression artérielle.
- Activation de la suppression de `diagnosed_diabetes` dans les colonnes de fuite de cible.
- Ajout d'un test vérifiant explicitement cette protection.
- Correction des annotations et des contrôles de valeur du module SHAP.

### Qualité et intégration continue

- Correction de toutes les erreurs Ruff détectées.
- Formatage homogène de l'ensemble du package canonique et des tests.
- Socle supporté relevé à Python 3.12–3.14 et MLflow 3.x.
- Déclenchement de la CI sur `main` et invalidation des caches lorsque `poetry.lock` change.

## Validation

- Tests : 100 réussis sous Python 3.12, avec un seuil minimal de couverture de 50 %.
- Ruff : aucune erreur.
- Formatage Ruff : conforme.
- Mypy : aucune erreur sur 48 fichiers sources.
- Compilation Python : réussie.
- Recherche de secrets codés en dur : aucun résultat.
- Audit des dépendances après mise à niveau : aucune vulnérabilité connue détectée.
- Construction Poetry : wheel et archive source générées avec succès.

## Industrialisation ajoutée

- Métadonnées modernes PEP 621 et marqueur de typage `py.typed`.
- Validation précoce des configurations d'entraînement et d'inférence.
- Validation des types, doublons, valeurs infinies et plages physiologiques.
- Workflow de publication PyPI par identité fédérée sur les tags de version.
- Tests CI de la wheel installée, audit des dépendances et mises à jour Dependabot.
- Pipeline Databricks Asset Bundle complet : ingestion, features, split, entraînement, évaluation, promotion puis inférence.
- Politiques de contribution, sécurité, licence et changelog.

## Points à surveiller

- Les artefacts MLflow utilisent `pickle`. Ils doivent être chargés uniquement depuis un stockage de modèles de confiance, car un fichier pickle hostile peut exécuter du code lors du chargement.
- Les seuils médicaux et les variables dérivées doivent être validés par un spécialiste clinique avant un usage réel sur des patients.
- Les intégrations XGBoost, CatBoost, LightGBM, SHAP, MLflow et Databricks nécessitent aussi des tests d'intégration dans l'environnement cible ; la suite locale couvre surtout le comportement unitaire.
