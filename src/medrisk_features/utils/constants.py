"""
Clinical constants used across the medrisk-features package.

All thresholds are sourced from established medical guidelines:
- ADA  : American Diabetes Association (glucose, HbA1c)
- WHO  : World Health Organization (BMI)
- JNC  : Joint National Committee (blood pressure)
- NCEP-ATP III : lipid and metabolic syndrome criteria
- WHO  : physical activity recommendations
"""

# ---------------------------------------------------------------------------
# Fasting glucose thresholds (ADA 2023, mg/dL)
# ---------------------------------------------------------------------------
GLUCOSE_NORMAL_MAX: int = 99          # ≤ 99  → Normal
GLUCOSE_PREDIABETES_MAX: int = 125    # 100–125 → Pre-Diabetes; ≥ 126 → Diabetes

# ---------------------------------------------------------------------------
# HbA1c thresholds (ADA 2023, %)
# ---------------------------------------------------------------------------
HBA1C_NORMAL_MAX: float = 5.7        # < 5.7  → Normal
HBA1C_PREDIABETES_MAX: float = 6.4   # 5.7–6.4 → Pre-Diabetes; ≥ 6.5 → Diabetes

# ---------------------------------------------------------------------------
# BMI thresholds (WHO, kg/m²)
# ---------------------------------------------------------------------------
BMI_UNDERWEIGHT_MAX: float = 18.5    # < 18.5 → Underweight
BMI_NORMAL_MAX: float = 24.9         # 18.5–24.9 → Normal
BMI_OVERWEIGHT_MAX: float = 29.9     # 25.0–29.9 → Overweight; ≥ 30 → Obese

# ---------------------------------------------------------------------------
# Blood pressure thresholds (JNC 7, mmHg)
# ---------------------------------------------------------------------------
BP_SYSTOLIC_NORMAL_MAX: int = 119        # < 120 systolic → Normal
BP_SYSTOLIC_PREHYPERTENSION_MAX: int = 139  # 120–139 → Pre-Hypertension
BP_DIASTOLIC_NORMAL_MAX: int = 79        # < 80 diastolic → Normal
BP_DIASTOLIC_PREHYPERTENSION_MAX: int = 89  # 80–89 → Pre-Hypertension

# ---------------------------------------------------------------------------
# HOMA-IR insulin resistance threshold
# ---------------------------------------------------------------------------
HOMA_IR_RESISTANCE_THRESHOLD: float = 2.5  # HOMA-IR > 2.5 → insulin resistant

# ---------------------------------------------------------------------------
# Lipid thresholds (NCEP-ATP III, mg/dL)
# ---------------------------------------------------------------------------
TRIGLYCERIDES_HIGH_THRESHOLD: int = 150  # ≥ 150 → elevated triglycerides
HDL_LOW_THRESHOLD: int = 40              # < 40 → low HDL (men); clinical cutoff

# ---------------------------------------------------------------------------
# Physical activity (WHO 2020 guidelines, min/week)
# ---------------------------------------------------------------------------
WHO_ACTIVITY_MIN_WEEKLY: int = 150  # ≥ 150 min/week moderate-intensity

# ---------------------------------------------------------------------------
# Lifestyle score
# ---------------------------------------------------------------------------
LIFESTYLE_SCORE_MAX: int = 10  # Maximum achievable lifestyle score (5 criteria × 2)
