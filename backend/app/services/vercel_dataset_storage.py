import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from app import config

SNAPSHOT_PATH = "port-congestion/active-dataset.zip"
MAX_SNAPSHOT_BYTES = 30 * 1024 * 1024
_UPLOAD_NAME = re.compile(r"uploads/[a-f0-9]{64}\.csv\Z")


class DatasetStorageError(OSError):
    pass


def _is_vercel() -> bool:
    return os.getenv("VERCEL") == "1"


def _blob_is_configured() -> bool:
    has_store = bool(os.getenv("BLOB_STORE_ID"))
    has_credentials = any(
        os.getenv(name)
        for name in (
            "VERCEL_OIDC_TOKEN",
            "BLOB_READ_WRITE_TOKEN",
            "VERCEL_BLOB_READ_WRITE_TOKEN",
        )
    )
    return (has_store and has_credentials) or bool(
        os.getenv("BLOB_READ_WRITE_TOKEN")
        or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN")
    )


def _blob_sdk():
    from vercel import blob

    return blob


def configure_serverless_dataset_storage() -> None:
    if not _is_vercel():
        return

    from app.delay_training import DELAY_MODEL_PATH
    from app.services import datasets, runtime

    package_paths = {
        "data": config.DATA_DIR,
        "processed": config.PROCESSED_PATH,
        "quality": config.QUALITY_PATH,
        "provenance": config.PROVENANCE_PATH,
    }
    root = Path(tempfile.mkdtemp(prefix="port-congestion-data-"))
    paths = {
        "data": root,
        "processed": root / "processed" / "port_daily.csv",
        "quality": root / "processed" / "data_quality.json",
        "provenance": root / "manifests" / "provenance.json",
    }
    paths["processed"].parent.mkdir(parents=True, exist_ok=True)
    paths["provenance"].parent.mkdir(parents=True, exist_ok=True)
    for key in ("processed", "quality", "provenance"):
        source = package_paths[key]
        if source.is_file():
            shutil.copy2(source, paths[key])

    config.DATA_DIR = paths["data"]
    config.PROCESSED_PATH = paths["processed"]
    config.QUALITY_PATH = paths["quality"]
    config.PROVENANCE_PATH = paths["provenance"]
    for module in (datasets, runtime):
        module.DATA_DIR = paths["data"]
        module.PROCESSED_PATH = paths["processed"]
        module.QUALITY_PATH = paths["quality"]
        module.PROVENANCE_PATH = paths["provenance"]
    from app import delay_training

    delay_training.PROCESSED_PATH = paths["processed"]
    delay_training.PROVENANCE_PATH = paths["provenance"]
    delay_training.DELAY_MODEL_PATH = DELAY_MODEL_PATH

    if not _blob_is_configured():
        return

    blob = _blob_sdk()
    try:
        snapshot = blob.sync.get_blob(SNAPSHOT_PATH, access="private").body
    except blob.BlobNotFoundError:
        return
    _restore_snapshot(snapshot, paths["data"])


def persist_dataset_snapshot(
    processed_path: Path,
    quality_path: Path,
    provenance_path: Path,
    upload_path: Path,
) -> None:
    if not _is_vercel():
        return
    if not _blob_is_configured():
        raise DatasetStorageError(
            "Dataset activation requires a connected private Vercel Blob store. "
            "Connect a store to this project and redeploy before uploading datasets."
        )

    provenance: dict[str, Any]
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    upload_name = str(provenance.get("uploaded_csv", ""))
    if not _UPLOAD_NAME.fullmatch(upload_name):
        raise DatasetStorageError("The uploaded dataset provenance has an invalid stored filename.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(processed_path, "processed/port_daily.csv")
        archive.write(quality_path, "processed/data_quality.json")
        archive.write(provenance_path, "manifests/provenance.json")
        archive.write(upload_path, upload_name)

    blob = _blob_sdk()
    try:
        blob.sync.put_blob(
            SNAPSHOT_PATH,
            buffer.getvalue(),
            access="private",
            content_type="application/zip",
            allow_overwrite=True,
        )
    except blob.BlobError as error:
        raise DatasetStorageError(
            "The dataset snapshot could not be saved to private Vercel Blob storage."
        ) from error


def _restore_snapshot(snapshot: bytes, data_dir: Path) -> None:
    if len(snapshot) > MAX_SNAPSHOT_BYTES:
        raise RuntimeError("The active Vercel Blob dataset snapshot exceeds the supported size.")
    expected = {
        "processed/port_daily.csv",
        "processed/data_quality.json",
        "manifests/provenance.json",
    }
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
        entries = archive.infolist()
        names = {entry.filename for entry in entries}
        if not expected.issubset(names) or any(
            name not in expected and not _UPLOAD_NAME.fullmatch(name)
            for name in names
        ):
            raise RuntimeError("The active Vercel Blob dataset snapshot has an invalid file layout.")
        if sum(entry.file_size for entry in entries) > MAX_SNAPSHOT_BYTES:
            raise RuntimeError("The active Vercel Blob dataset snapshot expands beyond the supported size.")
        for entry in entries:
            destination = data_dir / Path(entry.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(entry))
