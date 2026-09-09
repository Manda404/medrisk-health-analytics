from typing import Sequence

import pandas as pd
from pandas import DataFrame

from medrisk_health_analytics.features import (
    BehavioralFeatureEngineer,
    ClinicalFeatureEngineer,
    DemographicsFeatureEngineer,
    LifestyleFeatureEngineer,
    MedicalFeatureEngineer,
    MetabolicFeatureEngineer,
)
from medrisk_health_analytics.logging import get_logger
from medrisk_health_analytics.preprocessing import (
    clean_categorical_variables,
    drop_leakage_columns,
)
from medrisk_health_analytics.validation import DataSchemaValidator


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline for medical,
    metabolic and lifestyle risk modeling.

    Purpose
    -------
    Orchestrate all feature transformations in a
    reproducible, explainable and leakage-safe manner.
    """

    def __init__(
        self,
        age_group_strategy: str = "detailed",
        validate_schema: bool = True,
        logger=None,
    ):
        """
        Initialize the feature engineering pipeline.

        Parameters
        ----------
        age_group_strategy : str, default="detailed"
            Strategy used to bin age into categorical groups.
            - "detailed": fine-grained age bands (<30, 30–39, ..., 80+)
            - "coarse": broad categories (Young, Adult, Senior)

        validate_schema : bool, default=True
            Whether to validate the input dataset schema before
            applying feature engineering. When enabled, the pipeline
            checks for the presence of critical columns and raises
            explicit errors if they are missing.

        logger :
            Optional Loguru logger instance. If None, a default
            package logger is created and used.
        """
        self.logger = logger or get_logger(self.__class__.__name__)
        self.age_group_strategy = age_group_strategy
        self.validate_schema = validate_schema
        self.schema_validator = DataSchemaValidator(logger=self.logger)

        # Initialize feature blocks
        self.demographics = DemographicsFeatureEngineer(
            age_group_strategy=age_group_strategy,
            logger=self.logger,
        )
        self.medical = MedicalFeatureEngineer(logger=self.logger)
        self.clinical = ClinicalFeatureEngineer(logger=self.logger)
        self.metabolic = MetabolicFeatureEngineer(logger=self.logger)
        self.behavioral = BehavioralFeatureEngineer(logger=self.logger)
        self.lifestyle = LifestyleFeatureEngineer(logger=self.logger)

    def __getstate__(self) -> dict:
        """Return a pickle-safe representation for MLflow artifacts."""
        return {
            "age_group_strategy": self.age_group_strategy,
            "validate_schema": self.validate_schema,
        }

    def __setstate__(self, state: dict) -> None:
        """Rebuild runtime-only objects such as loggers after unpickling."""
        FeatureEngineeringPipeline.__init__(
            self,
            age_group_strategy=state.get("age_group_strategy", "detailed"),
            validate_schema=state.get("validate_schema", True),
        )

    def transform(
        self,
        df: DataFrame,
        *,
        target_column: str | None = None,
        id_columns: Sequence[str] = (),
    ) -> DataFrame:
        """
        Apply the full feature engineering pipeline.

        Parameters
        ----------
        df : DataFrame
            Raw input dataset.
        target_column : str or None
            Target preserved in the output and excluded from transformations.
        id_columns : sequence of str
            Identifier columns preserved without using them as features.

        Returns
        -------
        DataFrame
            Fully enriched dataset.
        """
        protected_columns = list(id_columns)
        if target_column is not None:
            protected_columns.append(target_column)
        missing = sorted(set(protected_columns) - set(df.columns))
        if missing:
            raise ValueError(f"Protected columns are missing: {missing}")

        feature_input = df.drop(columns=protected_columns)
        self.logger.info("Starting feature engineering pipeline...")
        df_enriched = feature_input.copy(deep=True)

        if self.validate_schema:
            self.schema_validator.validate(feature_input)
        # --------------------------------------------------
        # Step 0: Prevent data leakage
        # --------------------------------------------------
        df_enriched = drop_leakage_columns(df_enriched, logger=self.logger)

        # --------------------------------------------------
        # Step 1: Clean categorical variables
        # --------------------------------------------------
        df_enriched = clean_categorical_variables(
            df_enriched,
            logger=self.logger,
        )

        # --------------------------------------------------
        # Step 2: Demographics
        # --------------------------------------------------
        df_enriched = self.demographics.transform(df_enriched)

        # --------------------------------------------------
        # Step 3: Medical clinical features
        # --------------------------------------------------
        df_enriched = self.medical.transform(df_enriched)

        # --------------------------------------------------
        # Step 4: Clinical interactions
        # --------------------------------------------------
        df_enriched = self.clinical.transform(df_enriched)

        # --------------------------------------------------
        # Step 5: Advanced metabolism
        # --------------------------------------------------
        df_enriched = self.metabolic.transform(df_enriched)

        # --------------------------------------------------
        # Step 6: Behavioral features
        # --------------------------------------------------
        df_enriched = self.behavioral.transform(df_enriched)

        # --------------------------------------------------
        # Step 7: Lifestyle features
        # --------------------------------------------------
        df_enriched = self.lifestyle.transform(df_enriched)

        # Spark and MLflow model signatures handle plain Python string objects
        # more predictably than pandas CategoricalDtype columns.
        category_columns: list[str] = df_enriched.select_dtypes(
            include=["category"]
        ).columns.tolist()
        for column in category_columns:
            df_enriched[column] = df_enriched[column].astype(object)

        if protected_columns:
            preserved = df[protected_columns].reset_index(drop=True)
            df_enriched = pd.concat([preserved, df_enriched.reset_index(drop=True)], axis=1)

        self.logger.info(f"Pipeline completed successfully — total columns: {df_enriched.shape[1]}")
        return df_enriched
