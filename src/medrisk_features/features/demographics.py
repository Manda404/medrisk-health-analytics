import numpy as np
import pandas as pd
from pandas import DataFrame

from medrisk_features.logging import get_logger


class DemographicsFeatureEngineer:
    """
    Create demographic and socio-economic features.

    Purpose
    -------
    Capture age-related risk patterns and socio-economic vulnerability
    factors influencing metabolic and diabetic risk.

    Medical relevance
    ------------------
    Age and socio-economic context are major determinants
    of diabetes prevalence and health inequalities.
    """

    def __init__(
        self,
        age_group_strategy: str = "detailed",
        logger=None,
    ):
        """
        Parameters
        ----------
        age_group_strategy : str
            Age binning strategy.
            - "detailed": <30, 30–39, ..., 80+
            - "coarse":   Young, Adult, Senior
        """
        self.age_group_strategy = age_group_strategy
        self.logger = logger or get_logger(self.__class__.__name__)
        self._age_column = "Age"

    def _resolve_age_column(self, df: DataFrame) -> str:
        """Accept both historical `Age` and raw CSV `age` column names."""
        if "Age" in df.columns:
            return "Age"
        if "age" in df.columns:
            return "age"
        raise KeyError("Column 'Age' or 'age' is required for demographic features.")

    # ------------------------------------------------------------------
    # Age grouping strategies
    # ------------------------------------------------------------------
    def _create_age_group_detailed(self, df: DataFrame) -> DataFrame:
        bins = [0, 30, 40, 50, 60, 70, 80, np.inf]
        labels = ["<30", "30–39", "40–49", "50–59", "60–69", "70–79", "80+"]
        df["age_group"] = pd.cut(df[self._age_column], bins=bins, labels=labels, right=False)
        return df

    def _create_age_group_coarse(self, df: DataFrame) -> DataFrame:

        bins = [0, 30, 60, np.inf]
        labels = ["Young", "Adult", "Senior"]
        df["age_group"] = pd.cut(df[self._age_column], bins=bins, labels=labels, right=False)
        return df

    def _create_age_group(self, df: DataFrame) -> DataFrame:
        if self.age_group_strategy == "detailed":
            self.logger.info("Using detailed age grouping.")
            return self._create_age_group_detailed(df)

        if self.age_group_strategy == "coarse":
            self.logger.info("Using coarse age grouping.")
            return self._create_age_group_coarse(df)

        raise ValueError(
            f"Invalid age_group_strategy='{self.age_group_strategy}'. "
            "Valid options: ['detailed', 'coarse']"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply demographic feature engineering.

        Parameters
        ----------
        df : DataFrame
            Input dataset.

        Returns
        -------
        DataFrame
            Dataset enriched with demographic features.
        """
        df = df.copy()
        self.logger.info("Creating demographic features...")

        # 1. Age groups. Public datasets often use `age`; older package
        # examples used `Age`, so the transformer accepts both.
        self._age_column = self._resolve_age_column(df)

        df = self._create_age_group(df)

        # 2. Age non-linearity
        df["age_squared"] = df[self._age_column] ** 2

        # 3. Socio-economic vulnerability
        if {"income_level", "education_level"}.issubset(df.columns):
            df["socioeconomic_vulnerability_flag"] = (
                df["income_level"].isin(["Low", "Lower-Middle"])
                & df["education_level"].isin(["No formal", "Highschool"])
            ).astype(int)
        else:
            self.logger.warning(
                "Income or education columns missing — "
                "socioeconomic_vulnerability_flag not created."
            )

        self.logger.info("Demographic features created successfully.")
        return df
