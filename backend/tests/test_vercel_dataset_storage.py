import io
import importlib.util
import json
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import config
from app import delay_training
from app.main import app
from app.routes import api as api_routes
from app.services import datasets, runtime
from app.services import vercel_dataset_storage as storage


def test_private_blob_snapshot_round_trips_dataset_files(monkeypatch, tmp_path):
    processed = tmp_path / "port_daily.csv"
    quality = tmp_path / "data_quality.json"
    provenance = tmp_path / "provenance.json"
    upload = tmp_path / "uploaded.csv"
    processed.write_text("date,next_day_congestion\n2024-01-01,1\n", encoding="utf-8")
    quality.write_text('{"target_status":"provided"}', encoding="utf-8")
    provenance.write_text(
        json.dumps({"uploaded_csv": "uploads/" + "a" * 64 + ".csv"}),
        encoding="utf-8",
    )
    upload.write_text("MMSI,BaseDateTime\n", encoding="utf-8")
    stored = {}

    class FakeBlobSync:
        @staticmethod
        def put_blob(pathname, body, **kwargs):
            stored.update(pathname=pathname, body=body, options=kwargs)

    class FakeBlob:
        sync = FakeBlobSync()

    monkeypatch.setattr(storage, "_is_vercel", lambda: True)
    monkeypatch.setattr(storage, "_blob_is_configured", lambda: True)
    monkeypatch.setattr(storage, "_blob_sdk", lambda: FakeBlob)

    storage.persist_dataset_snapshot(processed, quality, provenance, upload)

    assert stored["pathname"] == storage.SNAPSHOT_PATH
    assert stored["options"]["access"] == "private"
    assert stored["options"]["allow_overwrite"] is True
    restore_dir = tmp_path / "restore"
    restore_dir.mkdir()
    storage._restore_snapshot(stored["body"], restore_dir)
    restored_processed = restore_dir / "processed" / "port_daily.csv"
    restored_quality = restore_dir / "processed" / "data_quality.json"
    restored_provenance = restore_dir / "manifests" / "provenance.json"
    restored_upload = restore_dir / ("uploads/" + "a" * 64 + ".csv")
    assert restored_processed.read_text(encoding="utf-8") == processed.read_text(encoding="utf-8")
    assert restored_quality.read_text(encoding="utf-8") == quality.read_text(encoding="utf-8")
    assert restored_provenance.read_text(encoding="utf-8") == provenance.read_text(encoding="utf-8")
    assert restored_upload.read_text(encoding="utf-8") == upload.read_text(encoding="utf-8")


def test_vercel_upload_requires_a_connected_blob_store(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_is_vercel", lambda: True)
    monkeypatch.setattr(storage, "_blob_is_configured", lambda: False)

    with pytest.raises(storage.DatasetStorageError, match="private Vercel Blob store"):
        storage.persist_dataset_snapshot(
            tmp_path / "processed.csv",
            tmp_path / "quality.json",
            tmp_path / "provenance.json",
            tmp_path / "uploaded.csv",
        )


def test_snapshot_restore_rejects_unexpected_archive_paths(tmp_path):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("../outside.txt", "not allowed")
        archive.writestr("processed/port_daily.csv", "date\n")
        archive.writestr("processed/data_quality.json", "{}")
        archive.writestr("manifests/provenance.json", "{}")

    with pytest.raises(RuntimeError, match="invalid file layout"):
        storage._restore_snapshot(output.getvalue(), tmp_path)
    assert not (tmp_path.parent / "outside.txt").exists()


def test_snapshot_restore_enforces_expanded_size_limit(monkeypatch, tmp_path):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("processed/port_daily.csv", "x" * 32)
        archive.writestr("processed/data_quality.json", "{}")
        archive.writestr("manifests/provenance.json", "{}")
    monkeypatch.setattr(storage, "MAX_SNAPSHOT_BYTES", 16)

    with pytest.raises(RuntimeError, match="exceeds the supported size"):
        storage._restore_snapshot(output.getvalue(), tmp_path)


def test_serverless_dataset_upload_limit_stays_below_vercel_body_cap(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    spec = importlib.util.spec_from_file_location(
        "vercel_upload_limit_probe", datasets.__file__
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.MAX_UPLOAD_BYTES == 4 * 1024 * 1024


def test_serverless_startup_restores_blob_snapshot_into_runtime_data_paths(
    monkeypatch, tmp_path
):
    bundled = tmp_path / "bundled"
    (bundled / "processed").mkdir(parents=True)
    (bundled / "manifests").mkdir()
    bundled_processed = bundled / "processed" / "port_daily.csv"
    bundled_quality = bundled / "processed" / "data_quality.json"
    bundled_provenance = bundled / "manifests" / "provenance.json"
    bundled_processed.write_text("date\n2024-01-01\n", encoding="utf-8")
    bundled_quality.write_text('{"source":"bundled"}', encoding="utf-8")
    bundled_provenance.write_text('{"source":"bundled"}', encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", bundled)
    monkeypatch.setattr(config, "PROCESSED_PATH", bundled_processed)
    monkeypatch.setattr(config, "QUALITY_PATH", bundled_quality)
    monkeypatch.setattr(config, "PROVENANCE_PATH", bundled_provenance)
    for module in (datasets, runtime):
        monkeypatch.setattr(module, "DATA_DIR", bundled)
        monkeypatch.setattr(module, "PROCESSED_PATH", bundled_processed)
        monkeypatch.setattr(module, "QUALITY_PATH", bundled_quality)
        monkeypatch.setattr(module, "PROVENANCE_PATH", bundled_provenance)
    monkeypatch.setattr(delay_training, "PROCESSED_PATH", bundled_processed)
    monkeypatch.setattr(delay_training, "PROVENANCE_PATH", bundled_provenance)
    runtime_root = tmp_path / "runtime-data"
    monkeypatch.setattr(storage.tempfile, "mkdtemp", lambda **kwargs: str(runtime_root))
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setattr(storage, "_blob_is_configured", lambda: True)

    snapshot_buffer = io.BytesIO()
    with zipfile.ZipFile(snapshot_buffer, "w") as archive:
        archive.writestr("processed/port_daily.csv", "date\n2025-01-01\n")
        archive.writestr("processed/data_quality.json", '{"source":"blob"}')
        archive.writestr("manifests/provenance.json", '{"source":"blob"}')

    class FakeBlobSync:
        @staticmethod
        def get_blob(pathname, *, access):
            assert pathname == storage.SNAPSHOT_PATH
            assert access == "private"
            return SimpleNamespace(body=snapshot_buffer.getvalue())

    class FakeBlob:
        sync = FakeBlobSync()
        BlobNotFoundError = type("BlobNotFoundError", (Exception,), {})

    monkeypatch.setattr(storage, "_blob_sdk", lambda: FakeBlob)

    storage.configure_serverless_dataset_storage()

    assert config.PROCESSED_PATH == runtime_root / "processed" / "port_daily.csv"
    assert datasets.PROCESSED_PATH == config.PROCESSED_PATH
    assert runtime.PROCESSED_PATH == config.PROCESSED_PATH
    assert config.PROCESSED_PATH.read_text(encoding="utf-8") == "date\n2025-01-01\n"
    assert config.QUALITY_PATH.read_text(encoding="utf-8") == '{"source":"blob"}'


def test_dataset_upload_returns_service_unavailable_when_blob_is_not_configured(
    monkeypatch,
):
    def reject_upload(*args, **kwargs):
        raise storage.DatasetStorageError("Connect a private Vercel Blob store.")

    monkeypatch.setattr(api_routes, "process_upload", reject_upload)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("dataset.csv", b"upload bytes", "text/csv")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Connect a private Vercel Blob store."
