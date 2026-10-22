import os
import shutil
import tempfile
from pathlib import Path

from app import config


def _seed_files(files: list[tuple[Path, Path]]) -> None:
    for source, destination in files:
        if destination.is_file():
            continue
        if not source.is_file():
            raise RuntimeError(
                f"Required deployment asset is missing: {source}. "
                "Include it in the repository or restore it to persistent storage."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as staged:
            staged_path = Path(staged.name)
        try:
            shutil.copy2(source, staged_path)
            os.replace(staged_path, destination)
        finally:
            staged_path.unlink(missing_ok=True)


def initialize_persistent_storage() -> None:
    bundled_data = config.ROOT_DIR / "data"
    data_files = [
        (bundled_data / "processed" / "port_daily.csv", config.PROCESSED_PATH),
        (bundled_data / "processed" / "data_quality.json", config.QUALITY_PATH),
        (bundled_data / "manifests" / "provenance.json", config.PROVENANCE_PATH),
    ]
    present = [destination.is_file() for _, destination in data_files]
    if any(present) and not all(present):
        raise RuntimeError(
            "Persistent dataset storage is partially initialized. Restore all of "
            "port_daily.csv, data_quality.json, and provenance.json before startup."
        )
    if not any(present):
        _seed_files(data_files)

    if os.getenv("PORT_ARTIFACTS_DIR"):
        bundled_artifacts = config.ROOT_DIR / "artifacts"
        artifact_files = [
            (
                bundled_artifacts / "models" / "congestion_model.joblib",
                config.MODEL_PATH,
            ),
            (
                bundled_artifacts / "models" / "delay_model.joblib",
                config.DELAY_MODEL_PATH,
            ),
            (
                bundled_artifacts / "metrics" / "evaluation.json",
                config.METRICS_PATH,
            ),
            (
                bundled_artifacts / "metrics" / "delay_evaluation.json",
                config.DELAY_METRICS_PATH,
            ),
        ]
        _seed_files(artifact_files)
