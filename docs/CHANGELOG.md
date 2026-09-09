# Changelog

## 0.4.1

- Aligned the Databricks input contract with the source dataset's `age` column.

## 0.4.0

- Added blocking data contracts before training and persisted quality evidence.
- Added feature drift monitoring with PSI and a dedicated Delta history table.
- Added MLflow dataset lineage, deployment environment, and training-source tags.
- Added a fourth Lakeflow task for post-inference monitoring.
- Added serverless environment-variable configuration for isolated bundle targets.
- Added a manual, environment-protected Databricks deployment workflow.
- Added the complete MLOps architecture and operations runbook.

This project follows Semantic Versioning and the Keep a Changelog format.

## Unreleased

### Changed

- Consolidated the source tree under `medrisk_health_analytics`.
- Migrated package metadata to PEP 621.
- Added strict configuration and input-schema validation.
- Added reproducible CI build, dependency audit, wheel smoke test, and trusted release workflow.
- Added an end-to-end Databricks Asset Bundle job definition.
- Replaced Databricks widgets with one validated `MEDRISK_*` runtime configuration.
- Migrated runtime settings to Pydantic and added rotating Loguru file output.
- Standardized Databricks installation on bundle-built wheel task libraries.
- Added Python 3.14 support and moved Databricks workloads to Runtime 18 LTS.
- Documented notebook responsibilities and secured model handoff between job tasks.
- Added a high-level API for loading, profiling, preprocessing, feature engineering, training, and prediction.
- Added class-based loaders, analyzers, preprocessors, trainers, and predictors.
- Added uniform wrappers for boosting, logistic regression, random forest, and gradient boosting.
- Consolidated the production workflow from seven notebooks to three.
- Added `ModelEvaluator` for metrics, diagnostics, feature importance, and quality gates.

### Fixed

- Included the `boosting.data` module in source control and built distributions.
- Prevented target leakage from `diagnosed_diabetes`.
- Fixed blood-pressure feature generation and SHAP typing failures.
