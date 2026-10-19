from __future__ import annotations

import json

import pytest

from tests.conftest import make_mini_frame
from wifi_tunja_smart_predictor.models.loaded_artifact import LoadedArtifact


@pytest.mark.parametrize("mismatch", ["classifier", "regressor"])
def test_evaluation_rejects_fingerprint_mismatch_before_writing_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path, mismatch: str
) -> None:
    import scripts.evaluate_model as evaluation

    classifier_artifact = LoadedArtifact(object(), "a" * 64, 10)
    regressor_artifact = LoadedArtifact(object(), "b" * 64, 12)
    classifier_hash = "c" * 64 if mismatch == "classifier" else "a" * 64
    regressor_hash = "c" * 64 if mismatch == "regressor" else "b" * 64
    classifier_metadata_path = tmp_path / "model_metadata.json"
    regressor_metadata_path = tmp_path / "regression_metadata.json"
    classifier_metadata_path.write_text(
        json.dumps({"artifact_sha256": classifier_hash, "artifact_size_bytes": 10}),
        encoding="utf-8",
    )
    regressor_metadata_path.write_text(
        json.dumps({"artifact_sha256": regressor_hash, "artifact_size_bytes": 12}),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "metrics.json"

    frame = make_mini_frame()
    monkeypatch.setattr(evaluation, "PROCESSED_DATASET_PATH", tmp_path / "dataset.csv")
    monkeypatch.setattr(evaluation, "load_raw_dataset", lambda _path: frame)
    monkeypatch.setattr(evaluation, "temporal_split", lambda data: (data, data, data))
    monkeypatch.setattr(evaluation, "MODEL_METADATA_PATH", classifier_metadata_path)
    monkeypatch.setattr(evaluation, "REGRESSION_METADATA_PATH", regressor_metadata_path)
    monkeypatch.setattr(evaluation, "load_model_artifact", lambda: classifier_artifact)
    monkeypatch.setattr(evaluation, "load_regression_artifact", lambda _path: regressor_artifact)
    monkeypatch.setattr(evaluation, "SELECTED_MODEL_METRICS_PATH", metrics_path)

    with pytest.raises(ValueError, match="fingerprint"):
        evaluation.main()

    assert not metrics_path.exists()
