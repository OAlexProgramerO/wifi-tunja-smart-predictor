from __future__ import annotations

from scripts import initialize_runtime


def test_initializer_runs_existing_pipeline_once_and_then_skips(tmp_path, capsys):
    assets = (tmp_path / ".runtime-ready", tmp_path / "dataset.csv", tmp_path / "model.joblib")
    calls = []

    def run_stage(command):
        calls.append(command[-1])
        if command[-1] == "scripts/train_model.py":
            for asset in assets[1:]:
                asset.touch()
        return 0

    assert (
        initialize_runtime.initialize_runtime(
            assets=assets, run_stage=run_stage, root=tmp_path, metrics_dir=tmp_path / "metrics"
        )
        == 0
    )
    assert calls == list(initialize_runtime.STAGES)
    assert "initialized successfully" in capsys.readouterr().out

    assert (
        initialize_runtime.initialize_runtime(
            assets=assets,
            run_stage=lambda _: (_ for _ in ()).throw(AssertionError("pipeline should not rerun")),
            root=tmp_path,
            metrics_dir=tmp_path / "metrics",
        )
        == 0
    )


def test_initializer_refuses_to_overwrite_partial_volumes(tmp_path, capsys):
    assets = (tmp_path / ".runtime-ready", tmp_path / "dataset.csv", tmp_path / "model.joblib")
    assets[1].touch()
    calls = []

    result = initialize_runtime.initialize_runtime(
        assets=assets,
        run_stage=lambda command: calls.append(command) or 0,
        root=tmp_path,
        metrics_dir=tmp_path / "metrics",
    )

    assert result == 1
    assert calls == []
    assert assets[1].is_file()
    assert "docker compose down -v" in capsys.readouterr().err


def test_initializer_stops_on_pipeline_failure(tmp_path, capsys):
    assets = (tmp_path / ".runtime-ready", tmp_path / "dataset.csv", tmp_path / "model.joblib")
    calls = []

    def fail_first_stage(command):
        calls.append(command[-1])
        return 1

    result = initialize_runtime.initialize_runtime(
        assets=assets,
        run_stage=fail_first_stage,
        root=tmp_path,
        metrics_dir=tmp_path / "metrics",
    )

    assert result == 1
    assert calls == [initialize_runtime.STAGES[0]]
    assert "failed in generate_synthetic_dataset.py" in capsys.readouterr().err
