import numpy as np
from pandas import DataFrame

from medrisk_health_analytics.logging import get_logger
from medrisk_health_analytics.utils.constants import (
    BMI_OVERWEIGHT_MAX,
    HDL_LOW_THRESHOLD,
    TRIGLYCERIDES_HIGH_THRESHOLD,
)


class MetabolicFeatureEngineer:
    """
    Create advanced metabolic features from glycemic, lipid and anthropometric biomarkers.

    Purpose
    -------
    Capture the global cardiometabolic burden associated with type 2 diabetes risk.

    Medical relevance
    ------------------
    Diabetes risk accumulates from combined imbalances:
    obesity, dyslipidemia, hypertension and chronic hyperglycemia.
    A composite burden score enables non-linear risk stratification.
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger(self.__class__.__name__)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply metabolic feature engineering.

        Parameters
        ----------
        df : DataFrame
            Input dataset.

        Returns
        -------
        DataFrame
            Dataset enriched with metabolic features.
        """
        df = df.copy()
        self.logger.info("Creating advanced metabolic features...")

        # --------------------------------------------------
        # 1) Triglyceride-glucose index: complementary insulin-resistance proxy
        # --------------------------------------------------
        if {"glucose_fasting", "triglycerides"}.issubset(df.columns):
            product = df["glucose_fasting"] * df["triglycerides"]
            df["tyg_index"] = np.log(product.where(product > 0) / 2)
            if "bmi" in df.columns:
                df["tyg_bmi_index"] = df["tyg_index"] * df["bmi"]
        else:
            self.logger.warning("Missing glucose or triglycerides — TyG features not created.")

        # --------------------------------------------------
        # 2) Dyslipidemia flag (NCEP-ATP III inspired)
        # --------------------------------------------------
        if {"triglycerides", "hdl_cholesterol"}.issubset(df.columns):
            df["dyslipidemia_flag"] = (
                (df["triglycerides"] >= TRIGLYCERIDES_HIGH_THRESHOLD)
                | (df["hdl_cholesterol"] < HDL_LOW_THRESHOLD)
            ).astype(int)
        else:
            self.logger.warning(
                "Missing triglycerides or hdl_cholesterol — dyslipidemia_flag not created."
            )

        # --------------------------------------------------
        # 3) Cardiometabolic burden score (0–5)
        # --------------------------------------------------
        required_score = {
            "bmi",
            "systolic_bp",
            "glucose_fasting",
            "triglycerides",
            "hdl_cholesterol",
        }
        if required_score.issubset(df.columns):
            df["cardiometabolic_burden"] = (
                (df["bmi"] >= BMI_OVERWEIGHT_MAX).astype(int)
                + (df["systolic_bp"] >= 130).astype(int)
                + (df["glucose_fasting"] >= 110).astype(int)
                + (df["triglycerides"] >= TRIGLYCERIDES_HIGH_THRESHOLD).astype(int)
                + (df["hdl_cholesterol"] < HDL_LOW_THRESHOLD).astype(int)
            )
        else:
            self.logger.warning(
                "Missing columns for cardiometabolic score — cardiometabolic_burden not created."
            )

        # --------------------------------------------------
        # 4) Blood pressure ratio (systolic / diastolic)
        # --------------------------------------------------
        if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
            df["blood_pressure_ratio"] = df["systolic_bp"] / df["diastolic_bp"].replace(0, np.nan)
        else:
            self.logger.warning("Missing BP columns — blood_pressure_ratio not created.")

        history_columns = {
            "family_history_diabetes",
            "hypertension_history",
            "cardiovascular_history",
        }
        if history_columns.issubset(df.columns):
            df["medical_history_burden"] = df[list(sorted(history_columns))].sum(axis=1)

        self.logger.info("Advanced metabolic features created successfully.")
        return df
