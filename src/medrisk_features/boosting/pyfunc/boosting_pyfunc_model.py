"""
BoostingPyFuncModel — the central mlflow.pyfunc.PythonModel for boosting.

This class is the single deployable unit that encapsulates:
  1. Preprocessing  (sklearn ColumnTransformer, serialized as preprocessor.pkl)
  2. Business rules (configurable priority thresholds)
  3. Trained model  (XGBoost / CatBoost / LightGBM, serialized natively)
  4. Prediction logic
  5. Post-processing (probability → priority label)
  6. Output formatting (client_id, pers_id, probability, priority, prediction_date, model_version)

Why pyfunc and not mlflow.xgboost / mlflow.catboost?
-----------------------------------------------------
Using mlflow.pyfunc.PythonModel as the base means:
  - a single unified interface regardless of the underlying library
  - the preprocessor travels WITH the model (no external dependency at inference)
  - the output format is guaranteed: any consumer gets the same clean DataFrame
  - the model can be loaded and run anywhere mlflow + the package are installed

Artifact layout (inside the MLflow run)
----------------------------------------
  boosting_model/
  ├── model.json (or .cbm / .txt depending on model type)
  ├── preprocessor.pkl
  ├── config.json
  └── feature_names.json

Usage (Databricks inference)
----------------------------
>>> import mlflow.pyfunc
>>> loaded = mlflow.pyfunc.load_model("models:/workspace.schema.model@Champion")
>>> df_out = loaded.predict(df_raw)
>>> display(df_out)
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd


class BoostingPyFuncModel:
    """
    MLflow PythonModel wrapper for XGBoost / CatBoost / LightGBM.

    Implements the full inference pipeline:
    raw input → preprocessing → boosting prediction → priority → clean output.

    Artifacts expected in context
    ------------------------------
    - "model"          : path to the serialized boosting model file
    - "preprocessor"   : path to preprocessor.pkl (sklearn ColumnTransformer)
    - "config"         : path to config.json (TrainingConfig fields)
    - "feature_names"  : path to feature_names.json (ordered list of feature cols)
    """

    # ── Artifact keys ─────────────────────────────────────────────────────────
    ARTIFACT_MODEL         = "model"
    ARTIFACT_PREPROCESSOR  = "preprocessor"
    ARTIFACT_CONFIG        = "config"
    ARTIFACT_FEATURE_NAMES = "feature_names"

    def load_context(self, context: Any) -> None:
        """
        Load all artifacts from the MLflow model context.

        Called automatically by MLflow when loading the model via
        mlflow.pyfunc.load_model(). Deserializes the boosting model,
        the preprocessor, the config, and the feature name list.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            Provides paths to serialized artifacts via context.artifacts.
        """
        # ── 1. Load configuration ─────────────────────────────────────────
        config_path = context.artifacts[self.ARTIFACT_CONFIG]
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: dict = json.load(f)

        # ── 2. Load feature names ─────────────────────────────────────────
        fn_path = context.artifacts[self.ARTIFACT_FEATURE_NAMES]
        with open(fn_path, "r", encoding="utf-8") as f:
            self.feature_names: list = json.load(f)

        # ── 3. Load the preprocessor (sklearn ColumnTransformer) ──────────
        preprocessor_path = context.artifacts[self.ARTIFACT_PREPROCESSOR]
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        # ── 4. Load the boosting model ────────────────────────────────────
        model_path = context.artifacts[self.ARTIFACT_MODEL]
        self._model = self._load_model(model_path)

        # ── 5. Store priority thresholds for fast access ──────────────────
        self._priority_high   = float(self.config.get("priority_high_threshold", 0.8))
        self._priority_medium = float(self.config.get("priority_medium_threshold", 0.5))
        self._id_columns      = self.config.get("id_columns", [])
        self._model_version   = self.config.get("model_version", "unknown")

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Full inference pipeline: raw data → clean output DataFrame.

        Steps
        -----
        1. Validate input type
        2. Extract identifier columns
        3. Select and order feature columns
        4. Apply sklearn preprocessor (imputation + encoding)
        5. Predict probability with the boosting model
        6. Apply priority business rules
        7. Return formatted output DataFrame

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            MLflow context (not used directly but required by the interface).
        model_input : pd.DataFrame
            Raw patient / client data. Must contain all feature columns.

        Returns
        -------
        pd.DataFrame
            Columns: <id_columns>, probability, priority, prediction_date,
            model_version.

        Raises
        ------
        ValueError
            If model_input is not a pandas DataFrame.
        KeyError
            If required feature columns are missing from model_input.
        """
        # ── Step 1: Validate input ────────────────────────────────────────
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError(
                f"model_input must be a pandas DataFrame, got {type(model_input).__name__}."
            )

        df = model_input.copy()

        # ── Step 2: Extract identifier columns ───────────────────────────
        id_data = pd.DataFrame()
        for col in self._id_columns:
            if col in df.columns:
                id_data[col] = df[col]

        # ── Step 3: Select and order feature columns ──────────────────────
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing feature columns in model_input: {missing}. "
                f"Expected: {self.feature_names}"
            )

        X = df[self.feature_names]

        # ── Step 4: Apply preprocessing ───────────────────────────────────
        X_transformed = self.preprocessor.transform(X)

        # Ensure output is a plain numpy array or DataFrame
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # ── Step 5: Predict probability ───────────────────────────────────
        probabilities = self._model.predict_proba(
            pd.DataFrame(X_transformed, columns=self.feature_names)
            if hasattr(self._model, "predict_proba")
            else X_transformed
        )

        # Flatten to 1D for binary classification
        if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
            probabilities = probabilities[:, 1]

        # ── Step 6: Apply priority business rules ─────────────────────────
        priority = self._compute_priority(probabilities)

        # ── Step 7: Build clean output DataFrame ──────────────────────────
        output = id_data.reset_index(drop=True).copy()
        output["probability"]      = np.round(probabilities, 6)
        output["priority"]         = priority
        output["prediction_date"]  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        output["model_version"]    = self._model_version

        return output

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_priority(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Map probabilities to priority labels using configurable thresholds.

        Priority rules
        --------------
        HIGH   : probability >= priority_high_threshold   (default 0.80)
        MEDIUM : probability >= priority_medium_threshold  (default 0.50)
        LOW    : otherwise

        Parameters
        ----------
        probabilities : np.ndarray  1D array of positive-class probabilities.

        Returns
        -------
        np.ndarray  1D array of string labels: 'HIGH', 'MEDIUM', 'LOW'.
        """
        conditions = [
            probabilities >= self._priority_high,
            probabilities >= self._priority_medium,
        ]
        choices = ["HIGH", "MEDIUM"]
        return np.select(conditions, choices, default="LOW")

    def _load_model(self, model_path: str) -> Any:
        """
        Deserialize the boosting model from disk.

        Detection strategy: check file extension to call the correct
        library's load method.

        Supported formats
        -----------------
        .json → XGBoost
        .cbm  → CatBoost
        .txt  → LightGBM
        .pkl  → any model serialized with pickle (fallback)
        """
        ext = os.path.splitext(model_path)[1].lower()

        if ext == ".json":
            try:
                import xgboost as xgb
                m = xgb.XGBClassifier()
                m.load_model(model_path)
                return m
            except ImportError as exc:
                raise ImportError(
                    "xgboost is required to load this model. "
                    "Install with: pip install medrisk-features[xgboost]"
                ) from exc

        elif ext == ".cbm":
            try:
                from catboost import CatBoostClassifier
                m = CatBoostClassifier()
                m.load_model(model_path)
                return m
            except ImportError as exc:
                raise ImportError(
                    "catboost is required to load this model. "
                    "Install with: pip install medrisk-features[catboost]"
                ) from exc

        elif ext == ".txt":
            try:
                import lightgbm as lgb
                return lgb.Booster(model_file=model_path)
            except ImportError as exc:
                raise ImportError(
                    "lightgbm is required to load this model. "
                    "Install with: pip install medrisk-features[lightgbm]"
                ) from exc

        else:
            # Generic pickle fallback (e.g. sklearn wrappers)
            with open(model_path, "rb") as f:
                return pickle.load(f)


def get_boosting_pyfunc_class():
    """
    Return BoostingPyFuncModel with mlflow.pyfunc.PythonModel as base,
    or the plain class if mlflow is not installed.

    This lazy pattern keeps mlflow optional for users who only use
    the feature engineering pipeline without MLflow.
    """
    try:
        import mlflow.pyfunc

        class _BoostingPyFuncModelWithBase(
            BoostingPyFuncModel,
            mlflow.pyfunc.PythonModel,
        ):
            """
            Production class: BoostingPyFuncModel registered as a
            first-class mlflow.pyfunc.PythonModel.
            """
            pass

        return _BoostingPyFuncModelWithBase

    except ImportError:
        return BoostingPyFuncModel
