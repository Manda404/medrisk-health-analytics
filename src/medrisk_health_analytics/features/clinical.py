import numpy as np
from pandas import DataFrame

from medrisk_health_analytics.logging import get_logger


class ClinicalFeatureEngineer:
    """
    Create clinical interaction features and lipid/glycemic ratios.

    Purpose
    -------
    Capture quantitative relationships between biomarkers
    that reflect metabolic balance and cardiovascular risk.

    Medical relevance
    ------------------
    Ratios such as HDL/LDL and cholesterol/HDL are strong
    indicators of cardiometabolic risk beyond raw values alone.
    Glycemic variability captures postprandial dysregulation.
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger(self.__class__.__name__)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply clinical interaction feature engineering.

        Parameters
        ----------
        df : DataFrame
            Input dataset.

        Returns
        -------
        DataFrame
            Dataset enriched with clinical interaction features.
        """
        df = df.copy()
        self.logger.info("Creating clinical interaction features...")

        # --------------------------------------------------
        # Lipid ratios (atherogenic risk markers)
        # --------------------------------------------------
        if {"hdl_cholesterol", "ldl_cholesterol"}.issubset(df.columns):
            # HDL/LDL ratio: higher is protective
            df["lipid_ratio_hdl_ldl"] = df["hdl_cholesterol"] / df["ldl_cholesterol"].replace(
                0, np.nan
            )
        else:
            self.logger.warning("HDL or LDL cholesterol missing — lipid_ratio_hdl_ldl not created.")

        if {"cholesterol_total", "hdl_cholesterol"}.issubset(df.columns):
            # Total cholesterol / HDL ratio: cardiovascular risk marker
            df["cholesterol_hdl_ratio"] = df["cholesterol_total"] / df["hdl_cholesterol"].replace(
                0, np.nan
            )
            df["non_hdl_cholesterol"] = (df["cholesterol_total"] - df["hdl_cholesterol"]).clip(
                lower=0
            )
        else:
            self.logger.warning(
                "Total cholesterol or HDL missing — cholesterol_hdl_ratio not created."
            )

        # --------------------------------------------------
        # BMI × glucose interaction (obesity-glycemia synergy)
        # --------------------------------------------------
        if {"bmi", "glucose_fasting"}.issubset(df.columns):
            df["bmi_glucose_interaction"] = df["bmi"] * df["glucose_fasting"]
        else:
            self.logger.warning(
                "BMI or fasting glucose missing — bmi_glucose_interaction not created."
            )

        # --------------------------------------------------
        # Glycemic variability (postprandial excursion)
        # --------------------------------------------------
        if {"glucose_postprandial", "glucose_fasting"}.issubset(df.columns):
            df["glucose_variability"] = df["glucose_postprandial"] - df["glucose_fasting"]
            df["glycemic_excursion_ratio"] = df["glucose_postprandial"] / df[
                "glucose_fasting"
            ].replace(0, np.nan)
        else:
            self.logger.warning(
                "Postprandial or fasting glucose missing — glucose_variability not created."
            )

        if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
            df["pulse_pressure"] = (df["systolic_bp"] - df["diastolic_bp"]).clip(lower=0)
            df["mean_arterial_pressure"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3

        if {"triglycerides", "hdl_cholesterol"}.issubset(df.columns):
            df["triglyceride_hdl_ratio"] = df["triglycerides"] / df["hdl_cholesterol"].replace(
                0, np.nan
            )

        self.logger.info("Clinical interaction features created successfully.")
        return df
