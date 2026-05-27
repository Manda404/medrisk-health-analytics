"""
Unit tests for MedRiskPyFuncModel and MLflow integration utilities.

These tests mock mlflow so they run without requiring a real MLflow server.
They verify:
  - MedRiskPyFuncModel.load_context() correctly deserializes artifacts
  - MedRiskPyFuncModel.predict() applies the pipeline and returns enriched data
  - PipelineLogResult dataclass holds the expected fields
  - load_pipeline raises ImportError when mlflow is not available
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from medrisk_features.mlflow.pyfunc_model import MedRiskPyFuncModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_mock(full_df):
    """Return a lightweight mock pipeline that calls the real transform."""
    from medrisk_features import FeatureEngineeringPipeline

    pipeline = FeatureEngineeringPipeline(validate_schema=False)
    return pipeline


def _build_artifacts(pipeline, tmp_dir: str) -> dict:
    """Serialize pipeline + config + feature_names to tmp_dir. Return artifact dict."""
    pipeline_path = os.path.join(tmp_dir, "pipeline.pkl")
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    config = {"age_group_strategy": "detailed", "validate_schema": False}
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    feature_names = ["Age", "glucose_fasting", "bmi", "glucose_category", "bmi_category"]
    fn_path = os.path.join(tmp_dir, "feature_names.json")
    with open(fn_path, "w") as f:
        json.dump(feature_names, f)

    return {
        "pipeline": pipeline_path,
        "config": config_path,
        "feature_names": fn_path,
    }


# ---------------------------------------------------------------------------
# Tests: MedRiskPyFuncModel
# ---------------------------------------------------------------------------

class TestMedRiskPyFuncModel:
    def test_load_context_deserializes_pipeline(self, full_df, logger):
        pipeline = _make_pipeline_mock(full_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = _build_artifacts(pipeline, tmpdir)

            # Simulate MLflow context
            context = MagicMock()
            context.artifacts = artifacts

            model = MedRiskPyFuncModel()
            model.load_context(context)

            assert model.pipeline is not None
            assert model.config["age_group_strategy"] == "detailed"
            assert isinstance(model.feature_names, list)

    def test_predict_returns_enriched_dataframe(self, full_df, logger):
        pipeline = _make_pipeline_mock(full_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = _build_artifacts(pipeline, tmpdir)
            context = MagicMock()
            context.artifacts = artifacts

            model = MedRiskPyFuncModel()
            model.load_context(context)

            df_out = model.predict(context, full_df)

            assert isinstance(df_out, pd.DataFrame)
            # The pipeline should add new columns
            assert df_out.shape[1] > full_df.shape[1]
            # Key engineered features should be present
            assert "glucose_category" in df_out.columns
            assert "bmi_category" in df_out.columns

    def test_predict_raises_on_non_dataframe(self, full_df):
        pipeline = _make_pipeline_mock(full_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = _build_artifacts(pipeline, tmpdir)
            context = MagicMock()
            context.artifacts = artifacts

            model = MedRiskPyFuncModel()
            model.load_context(context)

            with pytest.raises(ValueError, match="pandas DataFrame"):
                model.predict(context, [1, 2, 3])

    def test_predict_does_not_mutate_input(self, full_df):
        pipeline = _make_pipeline_mock(full_df)
        original_cols = set(full_df.columns)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = _build_artifacts(pipeline, tmpdir)
            context = MagicMock()
            context.artifacts = artifacts

            model = MedRiskPyFuncModel()
            model.load_context(context)
            model.predict(context, full_df)

            assert set(full_df.columns) == original_cols

    def test_load_context_handles_missing_feature_names_file(self, full_df):
        pipeline = _make_pipeline_mock(full_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = _build_artifacts(pipeline, tmpdir)
            # Remove feature_names from artifacts
            artifacts.pop("feature_names", None)
            context = MagicMock()
            context.artifacts = artifacts

            model = MedRiskPyFuncModel()
            model.load_context(context)

            # Should not crash; feature_names should be None
            assert model.feature_names is None


# ---------------------------------------------------------------------------
# Tests: PipelineLogResult dataclass
# ---------------------------------------------------------------------------

class TestPipelineLogResult:
    def test_dataclass_fields(self):
        from medrisk_features.mlflow.tracker import PipelineLogResult

        result = PipelineLogResult(
            run_id="abc123",
            model_uri="runs:/abc123/model",
            registered_model_name="workspace.schema.model",
            registered_model_version="2",
            feature_names=["col_a", "col_b"],
            config={"age_group_strategy": "detailed"},
        )

        assert result.run_id == "abc123"
        assert result.model_uri == "runs:/abc123/model"
        assert result.registered_model_name == "workspace.schema.model"
        assert result.registered_model_version == "2"
        assert len(result.feature_names) == 2

    def test_default_empty_collections(self):
        from medrisk_features.mlflow.tracker import PipelineLogResult

        result = PipelineLogResult(run_id="x", model_uri="runs:/x/m")
        assert result.feature_names == []
        assert result.config == {}
        assert result.artifact_paths == {}


# ---------------------------------------------------------------------------
# Tests: ImportError when mlflow is not available
# ---------------------------------------------------------------------------

class TestImportGuards:
    def test_load_pipeline_raises_import_error_without_mlflow(self):
        from medrisk_features.mlflow.tracker import load_pipeline

        with patch.dict("sys.modules", {"mlflow": None, "mlflow.pyfunc": None}):
            with pytest.raises(ImportError, match="mlflow"):
                load_pipeline("models:/workspace.schema.model@Champion")

    def test_log_pipeline_raises_import_error_without_mlflow(self, full_df):
        from medrisk_features.mlflow.tracker import log_pipeline

        pipeline = _make_pipeline_mock(full_df)
        with patch.dict("sys.modules", {"mlflow": None, "mlflow.pyfunc": None}):
            with pytest.raises(ImportError, match="mlflow"):
                log_pipeline(
                    pipeline=pipeline,
                    experiment_name="/test/experiment",
                )
