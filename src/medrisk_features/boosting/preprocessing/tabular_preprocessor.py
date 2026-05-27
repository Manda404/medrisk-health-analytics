"""
Tabular preprocessor for boosting models.

Wraps a sklearn ColumnTransformer that handles:
  - Numeric columns : median imputation + optional StandardScaler
  - Categorical cols: most_frequent imputation + OrdinalEncoder

The preprocessor is fitted on the training set and then serialized
alongside the model inside the MLflow artifact store.

Design note: tree-based models (XGBoost, CatBoost, LightGBM) do NOT
require feature scaling, so StandardScaler is disabled by default.
"""

from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np
import pandas as pd

# sklearn is an optional dependency — imported lazily inside methods
# so the module can be imported without sklearn being installed


class TabularPreprocessor:
    """
    Fit/transform preprocessor for tabular data with mixed column types.

    Parameters
    ----------
    numeric_columns : list of str
        Columns to treat as numeric.
    categorical_columns : list of str
        Columns to treat as categorical.
    numeric_impute_strategy : str
        Imputation strategy for numeric columns: 'mean', 'median', 'constant'.
    categorical_impute_strategy : str
        Imputation strategy for categorical columns: 'most_frequent', 'constant'.
    categorical_encoding : str
        'ordinal' (default, suitable for tree models) or 'onehot'.
    """

    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        categorical_encoding: str = "ordinal",
    ) -> None:
        self.numeric_columns            = numeric_columns or []
        self.categorical_columns        = categorical_columns or []
        self.numeric_impute_strategy    = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.categorical_encoding       = categorical_encoding

        self._transformer: Optional[ColumnTransformer] = None
        self._feature_names_out: List[str] = []
        self._is_fitted = False

    def _build_transformer(self):
        """Construct the ColumnTransformer from configuration."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder

        # ── Numeric pipeline ──────────────────────────────────────────────
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.numeric_impute_strategy)),
        ])

        # ── Categorical pipeline ──────────────────────────────────────────
        if self.categorical_encoding == "ordinal":
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            )
        else:
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.categorical_impute_strategy, fill_value="missing")),
            ("encoder", encoder),
        ])

        # ── Combine ───────────────────────────────────────────────────────
        transformers = []
        if self.numeric_columns:
            transformers.append(("numeric", numeric_pipeline, self.numeric_columns))
        if self.categorical_columns:
            transformers.append(("categorical", categorical_pipeline, self.categorical_columns))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        """
        Fit the preprocessor on training data.

        Parameters
        ----------
        X : pd.DataFrame  Training features (before target extraction).

        Returns
        -------
        self
        """
        self._transformer = self._build_transformer()
        self._transformer.fit(X)

        # Store output feature names for downstream use
        try:
            self._feature_names_out = self._transformer.get_feature_names_out().tolist()
        except Exception:
            self._feature_names_out = self.numeric_columns + self.categorical_columns

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted preprocessor to new data.

        Parameters
        ----------
        X : pd.DataFrame  Input features.

        Returns
        -------
        np.ndarray  Transformed feature array.
        """
        if not self._is_fitted or self._transformer is None:
            raise RuntimeError("TabularPreprocessor is not fitted. Call fit() first.")

        return self._transformer.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit then transform in one step."""
        return self.fit(X).transform(X)

    @property
    def feature_names_out(self) -> List[str]:
        """Output feature names after transformation."""
        return self._feature_names_out

    def save(self, path: str) -> None:
        """Serialize preprocessor to disk with pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "TabularPreprocessor":
        """Deserialize preprocessor from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_config(self) -> dict:
        """Return serializable configuration dict for MLflow logging."""
        return {
            "numeric_columns":             self.numeric_columns,
            "categorical_columns":         self.categorical_columns,
            "numeric_impute_strategy":     self.numeric_impute_strategy,
            "categorical_impute_strategy": self.categorical_impute_strategy,
            "categorical_encoding":        self.categorical_encoding,
        }


def auto_detect_column_types(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
    target: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
) -> tuple:
    """
    Auto-detect numeric and categorical columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame  Input DataFrame.
    exclude : list of str  Columns to explicitly exclude.
    target : str  Target column (excluded from features).
    id_columns : list of str  ID columns (excluded from features).

    Returns
    -------
    (numeric_cols, categorical_cols) : tuple of lists
    """
    skip = set(exclude or []) | set(id_columns or [])
    if target:
        skip.add(target)

    feature_cols = [c for c in df.columns if c not in skip]

    numeric_cols     = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    return numeric_cols, categorical_cols
