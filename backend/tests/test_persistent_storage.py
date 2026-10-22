import pytest

from app import config
from app.services.persistent_storage import initialize_persistent_storage


def _set_storage_paths(monkeypatch, data_dir, artifacts_dir, bundle_dir):
    monkeypatch.setattr(config, "ROOT_DIR", bundle_dir)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "PROCESSED_PATH", data_dir / "processed" / "port_daily.csv")
    monkeypatch.setattr(config, "QUALITY_PATH", data_dir / "processed" / "data_quality.json")
    monkeypatch.setattr(config, "PROVENANCE_PATH", data_dir / "manifests" / "provenance.json")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(config, "MODEL_PATH", artifacts_dir / "models" / "congestion_model.joblib")
    monkeypatch.setattr(config, "DELAY_MODEL_PATH", artifacts_dir / "models" / "delay_model.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", artifacts_dir / "metrics" / "evaluation.json")
    monkeypatch.setattr(
        config,
        "DELAY_METRICS_PATH",
        artifacts_dir / "metrics" / "delay_evaluation.json",
    )


def _write_bundled_files(root):
    files = {
        "data/processed/port_daily.csv": "date\n2024-01-01\n",
        "data/processed/data_quality.json": '{"quality":"bundled"}',
        "data/manifests/provenance.json": '{"provenance":"bundled"}',
        "artifacts/models/congestion_model.joblib": b"congestion model",
        "artifacts/models/delay_model.joblib": b"delay model",
        "artifacts/metrics/evaluation.json": '{"evaluation":"bundled"}',
        "artifacts/metrics/delay_evaluation.json": '{"delay_evaluation":"bundled"}',
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(contents, bytes) else "w"
        with path.open(mode, encoding=None if mode == "wb" else "utf-8") as handle:
            handle.write(contents)


def test_persistent_storage_seeds_bundle_assets_once_and_preserves_active_data(
    monkeypatch, tmp_path
):
    bundle = tmp_path / "bundle"
    persistent_data = tmp_path / "persistent" / "data"
    persistent_artifacts = tmp_path / "persistent" / "artifacts"
    _write_bundled_files(bundle)
    _set_storage_paths(monkeypatch, persistent_data, persistent_artifacts, bundle)
    monkeypatch.setenv("PORT_ARTIFACTS_DIR", str(persistent_artifacts))

    initialize_persistent_storage()

    assert config.PROCESSED_PATH.read_text(encoding="utf-8") == "date\n2024-01-01\n"
    assert config.PROVENANCE_PATH.is_file()
    assert config.MODEL_PATH.read_bytes() == b"congestion model"
    assert config.DELAY_MODEL_PATH.read_bytes() == b"delay model"
    assert config.METRICS_PATH.is_file()
    assert config.DELAY_METRICS_PATH.is_file()

    config.PROCESSED_PATH.write_text("date\n2025-01-01\n", encoding="utf-8")
    initialize_persistent_storage()

    assert config.PROCESSED_PATH.read_text(encoding="utf-8") == "date\n2025-01-01\n"


def test_persistent_storage_rejects_partially_initialized_dataset(
    monkeypatch, tmp_path
):
    bundle = tmp_path / "bundle"
    persistent_data = tmp_path / "persistent" / "data"
    persistent_artifacts = tmp_path / "persistent" / "artifacts"
    _write_bundled_files(bundle)
    _set_storage_paths(monkeypatch, persistent_data, persistent_artifacts, bundle)
    config.PROCESSED_PATH.parent.mkdir(parents=True)
    config.PROCESSED_PATH.write_text("date\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="partially initialized"):
        initialize_persistent_storage()
