from medrisk_features.utils.constants import (
    BMI_NORMAL_MAX,
    BMI_OVERWEIGHT_MAX,
    # BMI thresholds (WHO, kg/m²)
    BMI_UNDERWEIGHT_MAX,
    BP_DIASTOLIC_NORMAL_MAX,
    BP_DIASTOLIC_PREHYPERTENSION_MAX,
    # Blood pressure thresholds (JNC, mmHg)
    BP_SYSTOLIC_NORMAL_MAX,
    BP_SYSTOLIC_PREHYPERTENSION_MAX,
    # Glucose thresholds (ADA, mg/dL)
    GLUCOSE_NORMAL_MAX,
    GLUCOSE_PREDIABETES_MAX,
    # HbA1c thresholds (ADA, %)
    HBA1C_NORMAL_MAX,
    HBA1C_PREDIABETES_MAX,
    HDL_LOW_THRESHOLD,
    # HOMA-IR threshold
    HOMA_IR_RESISTANCE_THRESHOLD,
    # Lifestyle score
    LIFESTYLE_SCORE_MAX,
    # Lipid thresholds (NCEP-ATP III, mg/dL)
    TRIGLYCERIDES_HIGH_THRESHOLD,
    # Physical activity (WHO, min/week)
    WHO_ACTIVITY_MIN_WEEKLY,
)
from medrisk_features.utils.exceptions import (
    FeatureEngineeringError,
    InvalidConfigurationError,
    MedRiskError,
    MissingRequiredColumnError,
    SchemaValidationError,
)

__all__ = [
    "MedRiskError",
    "SchemaValidationError",
    "FeatureEngineeringError",
    "MissingRequiredColumnError",
    "InvalidConfigurationError",
    "GLUCOSE_NORMAL_MAX",
    "GLUCOSE_PREDIABETES_MAX",
    "HBA1C_NORMAL_MAX",
    "HBA1C_PREDIABETES_MAX",
    "BMI_UNDERWEIGHT_MAX",
    "BMI_NORMAL_MAX",
    "BMI_OVERWEIGHT_MAX",
    "BP_SYSTOLIC_NORMAL_MAX",
    "BP_SYSTOLIC_PREHYPERTENSION_MAX",
    "BP_DIASTOLIC_NORMAL_MAX",
    "BP_DIASTOLIC_PREHYPERTENSION_MAX",
    "HOMA_IR_RESISTANCE_THRESHOLD",
    "TRIGLYCERIDES_HIGH_THRESHOLD",
    "HDL_LOW_THRESHOLD",
    "WHO_ACTIVITY_MIN_WEEKLY",
    "LIFESTYLE_SCORE_MAX",
]
