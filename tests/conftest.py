import pandas as pd
import pytest
from medrisk_health_analytics.logging import get_logger


@pytest.fixture
def logger():
    return get_logger("test")


@pytest.fixture
def minimal_df():
    return pd.DataFrame(
        {
            "Age": [40, 60],
            "glucose_fasting": [110, 150],
            "bmi": [27, 32],
        }
    )


@pytest.fixture
def full_df():
    return pd.DataFrame(
        {
            "Age": [45, 62],
            "gender": ["Male", "Other"],
            "income_level": ["Low", "High"],
            "education_level": ["Highschool", "Bachelor"],
            "glucose_fasting": [110, 160],
            "hba1c": [6.1, 7.4],
            "bmi": [28, 34],
            "systolic_bp": [135, 150],
            "diastolic_bp": [85, 95],
            "triglycerides": [170, 210],
            "hdl_cholesterol": [38, 35],
            "ldl_cholesterol": [140, 160],
            "cholesterol_total": [210, 250],
            "insulin_level": [18, 30],
            "glucose_postprandial": [180, 240],
            "physical_activity_minutes_per_week": [120, 60],
            "screen_time_hours_per_day": [6, 8],
            "sleep_hours_per_day": [7, 6],
            "diet_score": [6, 4],
            "alcohol_consumption_per_week": [2, 5],
            "smoking_status": ["Former", "Current"],
        }
    )
