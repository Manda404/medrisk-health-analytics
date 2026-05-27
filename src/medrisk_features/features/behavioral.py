from pandas import DataFrame
from medrisk_features.logging import get_logger


class BehavioralFeatureEngineer:
    """
    Create behavioral features related to physical activity,
    sedentary exposure and sleep balance.

    Purpose
    -------
    Quantify lifestyle-related behavioral imbalances that
    directly impact insulin sensitivity and metabolic health.

    Medical relevance
    ------------------
    - WHO physical activity guidelines (≥150 min/week moderate intensity)
    - Screen overexposure is a proxy for sedentary behavior and circadian disruption
    - Sedentary behavior independently predicts type 2 diabetes risk
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger(self.__class__.__name__)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply behavioral feature engineering.

        Parameters
        ----------
        df : DataFrame
            Input dataset.

        Returns
        -------
        DataFrame
            Dataset enriched with behavioral features.
        """
        df = df.copy()
        self.logger.info("Creating behavioral features...")

        # --------------------------------------------------
        # 1) Physical activity adequacy (WHO threshold: 150 min/week)
        # --------------------------------------------------
        if "physical_activity_minutes_per_week" in df.columns:
            # Ratio relative to WHO recommendation, capped at 3x to avoid outlier distortion
            df["physical_activity_adequate"] = (
                df["physical_activity_minutes_per_week"] / 150
            ).clip(upper=3)
        else:
            self.logger.warning(
                "Column 'physical_activity_minutes_per_week' missing — "
                "physical_activity_adequate not created."
            )

        # --------------------------------------------------
        # 2) Screen / sleep imbalance (circadian disruption proxy)
        # --------------------------------------------------
        if {"screen_time_hours_per_day", "sleep_hours_per_day"}.issubset(df.columns):
            df["screen_sleep_imbalance"] = (
                df["screen_time_hours_per_day"]
                / df["sleep_hours_per_day"]
            ).clip(upper=5)
        else:
            self.logger.warning(
                "Screen time or sleep columns missing — screen_sleep_imbalance not created."
            )

        # --------------------------------------------------
        # 3) Sedentary risk flag (combined inactivity indicator)
        # --------------------------------------------------
        required = {
            "screen_time_hours_per_day",
            "physical_activity_minutes_per_week",
        }
        if required.issubset(df.columns):
            df["sedentary_risk"] = (
                (df["screen_time_hours_per_day"] >= 6)
                & (df["physical_activity_minutes_per_week"] < 150)
            ).astype(int)
        else:
            self.logger.warning(
                "Missing columns for sedentary_risk."
            )

        self.logger.info("Behavioral features created successfully.")
        return df
