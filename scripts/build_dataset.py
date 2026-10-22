import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from _provenance import safe_source_url

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.data_pipeline import build_daily_dataset, build_delay_dataset  # noqa: E402
from app.config import PROCESSED_PATH, PROVENANCE_PATH, QUALITY_PATH  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--source-url", default=None)
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    if "SyntheticDelayLabel" in raw.columns:
        daily, quality = build_delay_dataset(raw)
    else:
        daily, quality = build_daily_dataset(raw)
    if daily.empty:
        raise SystemExit("No daily observations remain after filtering; see validation and source scope.")
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    quality["source_dataset_sha256"] = input_hash
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(PROCESSED_PATH, index=False)
    quality.update(
        {
            "processed_rows": len(daily),
            "observation_start": daily["date"].min().date().isoformat(),
            "observation_end": daily["date"].max().date().isoformat(),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_PATH.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8")) if PROVENANCE_PATH.exists() else {}
    if provenance.get("sha256") != input_hash:
        source_url, sensitive_parts_redacted = safe_source_url(
            args.source_url or provenance.get("source_url")
        )
        source_name = args.source_name or provenance.get("source_name")
        verified_public_source = bool(source_url and source_url.startswith(("http://", "https://")) and source_name)
        provenance.update(
            {
                "data_status": "public" if verified_public_source else "unverified_import",
                "source_name": source_name or "Unverified CSV import",
                "source_url": source_url,
                "source_url_sensitive_parts_redacted": sensitive_parts_redacted,
                "filename": args.input.name,
                "file_size_bytes": args.input.stat().st_size,
                "sha256": input_hash,
                "original_columns": list(raw.columns),
                "acquisition_date": provenance.get("acquisition_date") or datetime.now(timezone.utc).date().isoformat(),
            }
        )
    provenance.update(
        {
            "original_columns": list(raw.columns),
            "processed_at": quality["processed_at"],
            "quality_report": str(QUALITY_PATH.relative_to(ROOT)),
            "processed_rows": int(len(daily)),
            "processed_observation_start": quality["observation_start"],
            "processed_observation_end": quality["observation_end"],
            "target_status": quality.get("target_status"),
            "target_source": quality.get("target_source"),
            "target_definition": quality.get("target_definition"),
            "delay_target_status": quality.get("delay_target_status"),
            "delay_target_source": quality.get("delay_target_source"),
            "delay_target_distribution": quality.get("delay_target_distribution"),
        }
    )
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Built {len(daily):,} daily rows at {PROCESSED_PATH}")
    print(f"Quality report: {QUALITY_PATH}")


if __name__ == "__main__":
    main()
