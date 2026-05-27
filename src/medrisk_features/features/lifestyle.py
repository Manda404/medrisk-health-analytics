import numpy as np
from pandas import DataFrame
from medrisk_features.logging import get_logger


class LifestyleFeatureEngineer:
    """
    Create lifestyle-related composite features.

    Purpose
    -------
    Summarize global lifestyle quality through
    interpretable and preventive indicators.

    Medical relevance
    ------------------
    Healthy lifestyle improves insulin sensitivity,
    reduces chronic inflammation and lowers diabetes risk.
    Each component is based on established prevention guidelines.
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Lifestyle score (0–10 scale, vectorized)
    # ------------------------------------------------------------------
    def _compute_lifestyle_score(self, df: DataFrame) -> DataFrame:
        required = {
            "diet_score",
            "physical_activity_minutes_per_week",
            "sleep_hours_per_day",
            "alcohol_consumption_per_week",
            "smoking_status",
        }

        if not required.issubset(df.columns):
            self.logger.warning(
                "Missing columns for lifestyle_score — score not created."
            )
            return df

        # Vectorized scoring — avoids slow row-by-row apply()
        # Each criterion contributes 2 points (max total = 10)
        score = (
            (df["diet_score"] >= 6).astype(int) * 2
            + (df["physical_activity_minutes_per_week"] >= 150).astype(int) * 2
            + ((df["sleep_hours_per_day"] >= 7) & (df["sleep_hours_per_day"] <= 9)).astype(int) * 2
            + (df["alcohol_consumption_per_week"] <= 2).astype(int) * 2
            + (df["smoking_status"] == "Never").astype(int) * 2
        )

        df["lifestyle_score"] = score
        return df

    # ------------------------------------------------------------------
    # Sleep efficiency (sleep quality relative to screen time)
    # ------------------------------------------------------------------
    def _compute_sleep_efficiency(self, df: DataFrame) -> DataFrame:
        if {"sleep_hours_per_day", "screen_time_hours_per_day"}.issubset(df.columns):
            # Ratio of sleep to (screen_time + 1) to avoid division by zero
            df["sleep_efficiency"] = (
                df["sleep_hours_per_day"]
                / (df["screen_time_hours_per_day"] + 1)
            ).clip(upper=2)
        else:
            self.logger.warning(
                "Missing sleep or screen time columns — sleep_efficiency not created."
            )
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply lifestyle feature engineering.

        Parameters
        ----------
        df : DataFrame
            Input dataset.

        Returns
        -------
        DataFrame
            Dataset enriched with lifestyle features.
        """
        df = df.copy()
        self.logger.info("Creating lifestyle features...")

        df = self._compute_lifestyle_score(df)
        df = self._compute_sleep_efficiency(df)

        self.logger.info("Lifestyle features created successfully.")
        return df
