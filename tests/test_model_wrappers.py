import numpy as np
import pandas as pd
import pytest

from medrisk_health_analytics import (
    GradientBoostingModel,
    LogisticRegressionModel,
    ModelFactory,
    RandomForestModel,
)


@pytest.fixture
def classification_data():
    X = pd.DataFrame(
        {
            "age": [25, 31, 38, 45, 52, 60, 67, 74],
            "bmi": [20, 23, 25, 27, 30, 32, 35, 38],
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


@pytest.mark.parametrize(
    "model_class,params",
    [
        (LogisticRegressionModel, {"max_iter": 200}),
        (RandomForestModel, {"n_estimators": 10, "n_jobs": 1}),
        (GradientBoostingModel, {"n_estimators": 10}),
    ],
)
def test_sklearn_wrappers_share_one_interface(model_class, params, classification_data):
    X, y = classification_data
    model = model_class(params=params).fit(X, y)

    assert model.predict(X).shape == (len(X),)
    assert model.predict_proba(X).shape == (len(X),)
    assert np.all((model.predict_proba(X) >= 0) & (model.predict_proba(X) <= 1))
    assert model.get_feature_importance().index.tolist() == ["bmi", "age"] or set(
        model.get_feature_importance().index
    ) == {"age", "bmi"}


def test_model_factory_creates_classical_models():
    assert isinstance(ModelFactory.create("logistic_regression"), LogisticRegressionModel)
    assert isinstance(ModelFactory.create("random_forest"), RandomForestModel)
    assert isinstance(ModelFactory.create("gradient_boosting"), GradientBoostingModel)


def test_sklearn_wrapper_save_and_load(tmp_path, classification_data):
    X, y = classification_data
    model = LogisticRegressionModel().fit(X, y)
    path = tmp_path / "logistic_model"

    model.save(str(path))
    loaded = LogisticRegressionModel.load(str(path))

    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
