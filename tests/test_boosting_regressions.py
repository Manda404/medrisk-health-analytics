import numpy as np
import pandas as pd
import pytest
from medrisk_features.boosting.models.factory import BoostingModelFactory
from medrisk_features.boosting.pyfunc.boosting_pyfunc_model import BoostingPyFuncModel


class DummyPreprocessor:
    feature_names_out = ["age", "gender_F", "gender_M"]

    def transform(self, X):
        return np.array(
            [
                [row.Age, 1, 0] if row.gender == "F" else [row.Age, 0, 1]
                for row in X.itertuples(index=False)
            ]
        )


class DummyClassifier:
    def predict_proba(self, X):
        assert X.columns.tolist() == ["age", "gender_F", "gender_M"]
        return np.array([[0.1, 0.9], [0.8, 0.2]])


def test_boosting_pyfunc_uses_transformed_feature_names_for_classification():
    model = BoostingPyFuncModel()
    model.feature_names = ["Age", "gender"]
    model.transformed_feature_names = ["age", "gender_F", "gender_M"]
    model.preprocessor = DummyPreprocessor()
    model._model = DummyClassifier()
    model._priority_high = 0.8
    model._priority_medium = 0.5
    model._id_columns = ["client_id"]
    model._model_version = "test"
    model._task_type = "binary_classification"

    df = pd.DataFrame(
        {
            "client_id": [1, 2],
            "Age": [50, 30],
            "gender": ["F", "M"],
        }
    )

    output = model.predict(None, df)

    assert output["client_id"].tolist() == [1, 2]
    assert output["probability"].tolist() == [0.9, 0.2]
    assert output["priority"].tolist() == ["HIGH", "LOW"]


def test_boosting_models_reject_regression_task_type():
    with pytest.raises(ValueError, match="Unsupported task_type"):
        BoostingModelFactory.create("xgboost", task_type="regression")
