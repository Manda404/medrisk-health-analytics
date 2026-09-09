import pandas as pd

from medrisk_health_analytics.mlflow.signature import make_nullable_safe_sample


def test_signature_sample_casts_integer_columns_without_mutating_input():
    original = pd.DataFrame(
        {
            "integer": pd.Series([1, 2], dtype="int64"),
            "nullable_integer": pd.Series([1, None], dtype="Int64"),
            "continuous": pd.Series([1.5, 2.5], dtype="float64"),
            "category": ["a", "b"],
        }
    )

    sample = make_nullable_safe_sample(original)

    assert sample["integer"].dtype == "float64"
    assert sample["nullable_integer"].dtype == "float64"
    assert sample["continuous"].dtype == "float64"
    assert sample["category"].dtype == "object"
    assert original["integer"].dtype == "int64"
    assert original["nullable_integer"].dtype == "Int64"
