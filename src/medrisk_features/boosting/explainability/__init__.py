"""
medrisk_features.boosting.explainability
=========================================
SHAP-based model explainability for XGBoost / CatBoost / LightGBM models.

Public API
----------
>>> from medrisk_features.boosting.explainability import BoostingShapExplainer
>>>
>>> explainer = BoostingShapExplainer(model, feature_names=feature_names)
>>> explainer.fit(X_train_df)
>>>
>>> fig = explainer.plot_summary(max_display=20)
>>> fig = explainer.plot_bar(max_display=20)
>>> fig = explainer.plot_waterfall(sample_idx=0)
>>> shap_df = explainer.get_shap_dataframe()

Requires: pip install medrisk-features[explainability]
"""

from medrisk_features.boosting.explainability.shap_explainer import (
    BoostingShapExplainer,
    ShapResult,
)

__all__ = [
    "BoostingShapExplainer",
    "ShapResult",
]
