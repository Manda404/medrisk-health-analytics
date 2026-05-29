"""
BoostingShapExplainer — SHAP analysis for XGBoost / CatBoost / LightGBM.

This module wraps the SHAP library's TreeExplainer behind a clean, unified
interface that is compatible with all three boosting backends supported by medrisk-health-analytics, and produces Matplotlib figures suitable for Databricks
notebooks (display(fig)) or local environments (fig.savefig / plt.show).

Three plot types are supported:

  1. Summary plot (beeswarm)  — global importance + direction per feature
  2. Bar plot                 — ranked mean |SHAP| per feature
  3. Waterfall plot           — per-sample decomposition of a single prediction

Why TreeExplainer?
------------------
shap.TreeExplainer is the fastest and most accurate SHAP explainer for
tree-based models. It exploits the tree structure directly to compute exact
SHAP values in polynomial time, without sampling approximations.

Installation
------------
pip install medrisk-health-analytics[explainability]   # adds shap + matplotlib

Usage (Databricks notebook)
----------------------------
>>> from medrisk_health_analytics.boosting.explainability import BoostingShapExplainer
>>>
>>> # After training:
>>> explainer = BoostingShapExplainer(model, feature_names=result.feature_names)
>>> explainer.fit(X_train_df)
>>>
>>> # Global view — which features matter most and in which direction?
>>> fig_summary = explainer.plot_summary(max_display=20, title="Feature Impact")
>>> display(fig_summary)
>>>
>>> # Ranked bar chart — top features by mean |SHAP|
>>> fig_bar = explainer.plot_bar(max_display=20)
>>> display(fig_bar)
>>>
>>> # Single-patient decomposition — why did this patient get HIGH priority?
>>> fig_wf = explainer.plot_waterfall(sample_idx=0)
>>> display(fig_wf)
>>>
>>> # Export SHAP values as a DataFrame for further analysis
>>> shap_df = explainer.get_shap_dataframe()
>>> display(shap_df.head())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from medrisk_health_analytics.boosting.models.base import BaseBoostingModel
from medrisk_health_analytics.logging.default_logger import get_logger

_logger = get_logger("medrisk-shap")

_SHAP_NOT_INSTALLED = (
    "shap is required for explainability features.\n"
    "Install with:  pip install medrisk-health-analytics[explainability]\n"
    "           or: pip install shap"
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ShapResult:
    """
    Container returned by BoostingShapExplainer.compute_shap_values().

    Attributes
    ----------
    shap_values : np.ndarray
        Raw SHAP values, shape (n_samples, n_features).
        For binary classification these are the SHAP values for the positive
        class (label = 1).
    feature_names : list of str
        Ordered feature names matching the columns of shap_values.
    expected_value : float
        Model base value (mean prediction in log-odds or probability space).
    X : pd.DataFrame
        Input DataFrame that was used to compute the SHAP values.
        Stored for downstream plotting.
    mean_abs_shap : pd.Series
        Mean absolute SHAP value per feature, sorted descending.
        Useful for quick feature ranking without plotting.
    """

    shap_values: np.ndarray
    feature_names: List[str]
    expected_value: float
    X: pd.DataFrame
    mean_abs_shap: pd.Series = field(default_factory=pd.Series)

    def __post_init__(self) -> None:
        if self.mean_abs_shap.empty:
            self.mean_abs_shap = (
                pd.Series(
                    np.abs(self.shap_values).mean(axis=0),
                    index=self.feature_names,
                    name="mean_abs_shap",
                ).sort_values(ascending=False)
            )

    def top_features(self, n: int = 10) -> pd.Series:
        """Return the top-n features by mean |SHAP| value."""
        return self.mean_abs_shap.head(n)


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------


class BoostingShapExplainer:
    """
    SHAP explainability wrapper for medrisk boosting models.

    Provides a single, consistent interface for computing and visualising
    SHAP (SHapley Additive exPlanations) values from XGBoost, CatBoost,
    or LightGBM models trained via the medrisk-health-analytics boosting pipeline.

    Parameters
    ----------
    model : BaseBoostingModel
        A fitted medrisk boosting model (XGBoostModel, CatBoostModel, or
        LightGBMModel). Must have been trained (model._is_fitted == True).
    feature_names : list of str or None
        Ordered list of feature column names. If None, falls back to the
        names stored in the model's internal _feature_names attribute.

    Examples
    --------
    >>> explainer = BoostingShapExplainer(result_model, feature_names=result.feature_names)
    >>> explainer.fit(X_train_preprocessed)
    >>> fig = explainer.plot_summary(max_display=20)
    >>> display(fig)   # Databricks
    """

    def __init__(
        self,
        model: BaseBoostingModel,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if not model._is_fitted:
            raise RuntimeError(
                "The boosting model must be fitted before creating a BoostingShapExplainer. "
                "Call train_boosting_model() or model.fit() first."
            )

        self._model = model
        self._feature_names: List[str] = (
            feature_names
            or getattr(model, "_feature_names", [])
            or []
        )
        self._explainer = None
        self._result: Optional[ShapResult] = None

    # ------------------------------------------------------------------
    # Fitting — build the TreeExplainer and compute SHAP values
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "BoostingShapExplainer":
        """
        Build the SHAP TreeExplainer and compute SHAP values for X.

        This method must be called before any plot method. It caches the
        computed SHAP values internally so subsequent plot calls are instant.

        Parameters
        ----------
        X : pd.DataFrame
            Preprocessed feature DataFrame (same format as used during
            training — after TabularPreprocessor.transform()).
            Typically use a representative sample (e.g. X_train or X_valid).
            Rows with NaN are silently dropped before computing SHAP values.

        Returns
        -------
        self
            Returns the explainer so calls can be chained:
            ``explainer.fit(X).plot_summary()``
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError(_SHAP_NOT_INSTALLED) from exc

        _logger.info(f"Building SHAP TreeExplainer for {self._model.model_name}…")

        # Drop rows with NaN — TreeExplainer does not handle NaN
        X_clean = X.dropna()
        if len(X_clean) < len(X):
            dropped = len(X) - len(X_clean)
            warnings.warn(
                f"BoostingShapExplainer.fit(): dropped {dropped} rows with NaN values "
                f"before computing SHAP values.",
                UserWarning,
                stacklevel=2,
            )

        # Reconcile feature names from the DataFrame columns
        if not self._feature_names:
            self._feature_names = X_clean.columns.tolist()

        # Build TreeExplainer on the underlying library model object
        raw_model = self._model._model
        try:
            self._explainer = shap.TreeExplainer(raw_model)
        except Exception as exc:
            raise RuntimeError(
                f"Could not build a SHAP TreeExplainer for "
                f"{type(raw_model).__name__}. Error: {exc}"
            ) from exc

        _logger.info(f"Computing SHAP values on {len(X_clean):,} samples…")

        # Compute raw SHAP values (ndarray for TreeExplainer)
        raw_shap = self._explainer.shap_values(X_clean)

        # For binary classification some backends return a list [neg, pos]
        # We want the SHAP values for the positive class (index 1)
        if isinstance(raw_shap, list):
            if len(raw_shap) == 2:
                shap_values = raw_shap[1]
            else:
                # Multiclass: use class 1 as default; user can override via raw access
                shap_values = raw_shap[1]
                warnings.warn(
                    "Multiclass model detected. plot_summary / plot_bar show SHAP values "
                    "for class index 1. Use explainer.result.shap_values for full access.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            shap_values = raw_shap

        # Scalar or array expected value
        ev = self._explainer.expected_value
        expected_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

        self._result = ShapResult(
            shap_values=shap_values,
            feature_names=self._feature_names,
            expected_value=expected_value,
            X=X_clean.reset_index(drop=True),
        )

        _logger.info(
            f"SHAP computation complete — "
            f"top feature: '{self._result.mean_abs_shap.index[0]}' "
            f"(mean |SHAP| = {self._result.mean_abs_shap.iloc[0]:.4f})"
        )
        return self

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    @property
    def result(self) -> ShapResult:
        """Return the cached ShapResult. Call fit() first."""
        self._check_fitted()
        return self._result  # type: ignore[return-value]

    def get_shap_dataframe(self) -> pd.DataFrame:
        """
        Return SHAP values as a tidy DataFrame.

        Each row corresponds to one sample, each column to one feature.
        Values represent the SHAP contribution of that feature to the model
        output for that sample (positive = pushes toward class 1).

        Returns
        -------
        pd.DataFrame
            Shape (n_samples, n_features), columns = feature names.

        Example
        -------
        >>> shap_df = explainer.get_shap_dataframe()
        >>> shap_df.describe()                        # distribution per feature
        >>> shap_df["bmi_category"].sort_values()     # impact of BMI on all patients
        """
        self._check_fitted()
        return pd.DataFrame(
            self._result.shap_values,  # type: ignore[union-attr]
            columns=self._feature_names,
        )

    def top_features(self, n: int = 10) -> pd.Series:
        """Return the top-n features by mean absolute SHAP value (descending)."""
        self._check_fitted()
        return self._result.top_features(n)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Plot 1 — Summary (beeswarm)
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        max_display: int = 20,
        title: str = "SHAP Feature Impact — Summary",
        figsize: tuple = (10, 7),
        alpha: float = 0.6,
        color_bar: bool = True,
    ) -> "matplotlib.figure.Figure":
        """
        Beeswarm summary plot: global importance + direction per feature.

        Each dot represents one sample. The x-axis shows the SHAP value
        (positive = pushes toward high risk / class 1; negative = toward
        low risk / class 0). Colour encodes the original feature value
        (red = high, blue = low).

        This is the most informative single plot — it shows both WHICH
        features matter and WHETHER high values of a feature increase or
        decrease the predicted risk.

        Parameters
        ----------
        max_display : int, default 20
            Number of top features to show (by mean |SHAP|).
        title : str
            Plot title shown above the figure.
        figsize : tuple of (width, height), default (10, 7)
            Matplotlib figure size in inches.
        alpha : float, default 0.6
            Point transparency (0 = invisible, 1 = opaque).
        color_bar : bool, default True
            Whether to show the feature-value colour bar on the right.

        Returns
        -------
        matplotlib.figure.Figure
            The Matplotlib figure. In Databricks: ``display(fig)``.
            Locally: ``fig.savefig("summary.png")`` or ``plt.show()``.

        Example
        -------
        >>> fig = explainer.plot_summary(max_display=20)
        >>> display(fig)          # Databricks
        >>> fig.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import shap

        self._check_fitted()
        result = self._result  # type: ignore[union-attr]

        fig, ax = plt.subplots(figsize=figsize)
        plt.sca(ax)

        shap.summary_plot(
            result.shap_values,
            result.X,
            feature_names=self._feature_names,
            max_display=max_display,
            plot_type="dot",
            alpha=alpha,
            color_bar=color_bar,
            show=False,
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("SHAP value  (impact on model output)", fontsize=10)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Plot 2 — Bar (mean |SHAP|)
    # ------------------------------------------------------------------

    def plot_bar(
        self,
        max_display: int = 20,
        title: str = "SHAP Feature Importance — Mean |SHAP|",
        figsize: tuple = (9, 6),
        color: str = "#2196F3",
    ) -> "matplotlib.figure.Figure":
        """
        Horizontal bar chart ranking features by mean absolute SHAP value.

        Unlike the built-in `get_feature_importance()` (which uses Gini
        impurity or gain), mean |SHAP| measures the average impact on the
        model output in the same unit as the prediction, making it directly
        comparable across different model types.

        Parameters
        ----------
        max_display : int, default 20
            Number of top features to display.
        title : str
            Plot title shown above the figure.
        figsize : tuple of (width, height), default (9, 6)
            Matplotlib figure size in inches.
        color : str, default "#2196F3" (blue)
            Bar colour (any valid Matplotlib colour string or hex code).

        Returns
        -------
        matplotlib.figure.Figure

        Example
        -------
        >>> fig = explainer.plot_bar(max_display=15, color="#E91E63")
        >>> display(fig)
        """
        import matplotlib.pyplot as plt

        self._check_fitted()
        result = self._result  # type: ignore[union-attr]

        top = result.mean_abs_shap.head(max_display).iloc[::-1]  # reverse for horizontal bar

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(top.index, top.values, color=color, edgecolor="white", linewidth=0.5)

        # Add value labels at the end of each bar
        for bar, val in zip(bars, top.values):
            ax.text(
                bar.get_width() + top.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#444444",
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_ylabel("Feature", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Plot 3 — Waterfall (single prediction)
    # ------------------------------------------------------------------

    def plot_waterfall(
        self,
        sample_idx: int = 0,
        max_display: int = 15,
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
    ) -> "matplotlib.figure.Figure":
        """
        Waterfall decomposition for a single patient prediction.

        Shows how each feature's SHAP value pushes the prediction up (red)
        or down (blue) from the model's base value (expected_value) to the
        final output for the selected sample.

        This is the key tool for explaining an individual prediction to a
        clinician or auditor: "This patient was classified HIGH risk because
        their glucose was 180 mg/dL (+0.32) and their BMI was 36 (+0.18)…"

        Parameters
        ----------
        sample_idx : int, default 0
            Row index in the DataFrame passed to fit() to explain.
            Use 0 for the first patient, 1 for the second, etc.
        max_display : int, default 15
            Maximum number of features to show in the waterfall.
            Remaining features are grouped into an "other features" bar.
        title : str or None
            Plot title. Defaults to "SHAP Waterfall — Sample #{sample_idx}".
        figsize : tuple of (width, height), default (10, 6)
            Matplotlib figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        IndexError
            If sample_idx is out of range of the fitted dataset.

        Example
        -------
        >>> # Explain the patient at row 42 of the training set
        >>> fig = explainer.plot_waterfall(sample_idx=42, max_display=12)
        >>> display(fig)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        self._check_fitted()
        result = self._result  # type: ignore[union-attr]
        n_samples = result.shap_values.shape[0]

        if sample_idx < 0 or sample_idx >= n_samples:
            raise IndexError(
                f"sample_idx={sample_idx} is out of range. "
                f"The explainer was fitted on {n_samples} samples (indices 0–{n_samples-1})."
            )

        shap_vals = result.shap_values[sample_idx]
        feature_vals = result.X.iloc[sample_idx]
        base_value = result.expected_value
        title = title or f"SHAP Waterfall — Sample #{sample_idx}"

        # Sort features by |SHAP| descending, keep top max_display
        order = np.argsort(np.abs(shap_vals))[::-1]
        top_idx = order[:max_display]
        other_idx = order[max_display:]

        top_shap = shap_vals[top_idx]
        top_names = [self._feature_names[i] for i in top_idx]
        top_fvals = [feature_vals.iloc[i] for i in top_idx]

        # Aggregate "other" features
        other_shap = shap_vals[other_idx].sum() if len(other_idx) > 0 else 0.0

        # Build the waterfall data (bottom to top: other → feature1 → … → featureN)
        names_plot = []
        shap_plot = []

        if len(other_idx) > 0:
            names_plot.append(f"{len(other_idx)} other features")
            shap_plot.append(other_shap)

        # Reverse so most important is at the top
        for name, sv, fv in zip(top_names[::-1], top_shap[::-1], top_fvals[::-1]):
            # Format feature value nicely
            if isinstance(fv, float):
                label = f"{name} = {fv:.3g}"
            else:
                label = f"{name} = {fv}"
            names_plot.append(label)
            shap_plot.append(sv)

        shap_arr = np.array(shap_plot)
        n_bars = len(shap_arr)

        # Compute cumulative positions for the waterfall
        cumulative = np.zeros(n_bars + 1)
        cumulative[0] = base_value
        for i, sv in enumerate(shap_arr):
            cumulative[i + 1] = cumulative[i] + sv
        final_value = cumulative[-1]

        # Colors
        colors = ["#EF5350" if sv >= 0 else "#42A5F5" for sv in shap_arr]

        fig, ax = plt.subplots(figsize=figsize)
        y_positions = np.arange(n_bars)

        for i, (sv, col) in enumerate(zip(shap_arr, colors)):
            left = min(cumulative[i], cumulative[i + 1])
            width = abs(sv)
            ax.barh(y_positions[i], width, left=left, color=col, edgecolor="white", linewidth=0.5)
            # SHAP value annotation
            x_text = cumulative[i + 1] + (0.003 if sv >= 0 else -0.003)
            ha = "left" if sv >= 0 else "right"
            sign = "+" if sv >= 0 else ""
            ax.text(x_text, y_positions[i], f"{sign}{sv:.4f}",
                    va="center", ha=ha, fontsize=8, color="#333333")

        # Base value line
        ax.axvline(x=base_value, color="#9E9E9E", linewidth=1.2,
                   linestyle="--", label=f"Base value = {base_value:.4f}")

        # Final value line
        ax.axvline(x=final_value, color="#212121", linewidth=1.5,
                   linestyle="-", label=f"f(x) = {final_value:.4f}")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names_plot, fontsize=8.5)
        ax.set_xlabel("Model output (SHAP contribution)", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend
        pos_patch = mpatches.Patch(color="#EF5350", label="Increases risk (↑)")
        neg_patch = mpatches.Patch(color="#42A5F5", label="Decreases risk (↓)")
        ax.legend(
            handles=[pos_patch, neg_patch],
            loc="lower right",
            fontsize=8.5,
            framealpha=0.85,
        )

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise RuntimeError if fit() has not been called."""
        if self._result is None:
            raise RuntimeError(
                "BoostingShapExplainer has not been fitted yet. "
                "Call explainer.fit(X) before plotting or accessing results."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._result is not None else "not fitted"
        n_features = len(self._feature_names)
        return (
            f"BoostingShapExplainer("
            f"model={self._model.model_name}, "
            f"n_features={n_features}, "
            f"status={status})"
        )
