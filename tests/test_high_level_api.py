import numpy as np
import pandas as pd
import pytest

from medrisk_health_analytics import (
    DatasetAnalyzer,
    DatasetLoader,
    DatasetPreprocessor,
    FeatureEngineeringPipeline,
    ModelPredictor,
    analyze_dataset,
    build_features,
    load_dataset,
    preprocess_dataset,
)


def test_load_and_analyze_dataframe():
    source = pd.DataFrame({"target": [0, 1, 1], "value": [1.0, None, 3.0]})

    loaded = load_dataset(source)
    analysis = analyze_dataset(loaded, target_column="target")

    assert loaded is not source
    assert analysis.rows == 3
    assert analysis.missing_values["value"] == 1
    assert analysis.target_distribution == {1: 2, 0: 1}


def test_preprocess_dataset_normalizes_without_mutating_input():
    source = pd.DataFrame(
        {" value ": [1.0, np.inf, 1.0], "smoking_status": ["Former", "Never", "Former"]}
    )

    result = preprocess_dataset(source)

    assert list(source.columns) == [" value ", "smoking_status"]
    assert list(result.columns) == ["value", "smoking_status"]
    assert len(result) == 2
    assert pd.isna(result.loc[1, "value"])
    assert result.loc[0, "smoking_status"] == "Ex-Smoker"


def test_build_features_preserves_target_and_ids(full_df):
    source = full_df.copy()
    source.insert(0, "patient_id", ["P001", "P002"])
    source["target"] = [1, 0]

    result = build_features(
        source,
        target_column="target",
        id_columns=["patient_id"],
        pipeline=FeatureEngineeringPipeline(validate_schema=True),
    )

    assert result.loc[0, "patient_id"] == "P001"
    assert result.loc[0, "target"] == 1
    assert "age_group" in result.columns


def test_analyze_dataset_rejects_missing_target():
    with pytest.raises(ValueError, match="Target column"):
        analyze_dataset(pd.DataFrame({"value": [1]}), target_column="missing")


def test_class_based_data_workflow():
    source = pd.DataFrame({"target": [0, 1], "value": [1.0, np.inf]})

    loaded = DatasetLoader().load(source)
    cleaned = DatasetPreprocessor().transform(loaded)
    analysis = DatasetAnalyzer(target_column="target").analyze(cleaned)

    assert analysis.rows == 2
    assert pd.isna(cleaned.loc[1, "value"])


def test_model_predictor_rejects_empty_uri():
    with pytest.raises(ValueError, match="model_uri"):
        ModelPredictor("")
