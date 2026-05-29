import pandas as pd
import pytest
from medrisk_health_analytics.boosting import split_dataframe


def test_split_dataframe_stratifies_pandas_target_distribution():
    df = pd.DataFrame(
        {
            "patient_id": range(100),
            "feature": range(100),
            "TARGET": [0] * 80 + [1] * 20,
        }
    )

    result = split_dataframe(
        df,
        target_column="TARGET",
        test_size=0.25,
        random_state=42,
        stratify=True,
    )

    assert result.train_count == 75
    assert result.test_count == 25
    assert result.train_target_distribution == {"0": 60, "1": 15}
    assert result.test_target_distribution == {"0": 20, "1": 5}
    assert set(result.train_df.columns) == set(df.columns)
    assert set(result.test_df.columns) == set(df.columns)


def test_split_dataframe_can_disable_stratification():
    df = pd.DataFrame(
        {
            "patient_id": range(20),
            "feature": range(20),
            "TARGET": [0] * 10 + [1] * 10,
        }
    )

    result = split_dataframe(
        df,
        target_column="TARGET",
        test_size=0.2,
        random_state=42,
        stratify=False,
    )

    assert result.train_count == 16
    assert result.test_count == 4


def test_split_dataframe_rejects_missing_target():
    df = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(KeyError, match="TARGET"):
        split_dataframe(df, target_column="TARGET")


def test_split_dataframe_rejects_invalid_test_size():
    df = pd.DataFrame({"feature": [1, 2, 3], "TARGET": [0, 1, 0]})

    with pytest.raises(ValueError, match="test_size"):
        split_dataframe(df, target_column="TARGET", test_size=1.0)


def test_split_dataframe_with_patient_csv_if_available():
    path = "data/patient.csv"
    pytest.importorskip("sklearn")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        pytest.skip("Local patient.csv fixture is not available.")

    result = split_dataframe(
        df,
        target_column="diagnosed_diabetes",
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    assert result.train_count == 80000
    assert result.test_count == 20000
    assert result.train_target_distribution == {"0": 32002, "1": 47998}
    assert result.test_target_distribution == {"0": 8000, "1": 12000}
