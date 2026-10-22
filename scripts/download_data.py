import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from _provenance import safe_source_url

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifests" / "provenance.json"
SOURCE_PAGE = "https://marinecadastre.gov/ais/"


def main():
    parser = argparse.ArgumentParser(description="Import a verified AIS CSV export.")
    parser.add_argument("--url", required=True, help="Actual downloadable URL copied from the source portal.")
    parser.add_argument("--filename", required=True, help="CSV filename to save in data/raw.")
    parser.add_argument("--source-name", required=True, help="Exact dataset or export title from the source portal.")
    args = parser.parse_args()
    filename = Path(args.filename).name
    if not filename.lower().endswith(".csv"):
        parser.error("--filename must end with .csv")
    destination = RAW_DIR / filename
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.url.startswith("file://"):
        shutil.copyfile(Path(args.url.removeprefix("file://")), destination)
    else:
        request = urllib.request.Request(args.url, headers={"User-Agent": "port-congestion-local-pipeline/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    if destination.stat().st_size == 0:
        destination.unlink()
        raise ValueError("Downloaded file is empty.")
    safe_url, redacted = safe_source_url(args.url)
    manifest = {
        "data_status": "public_import_pending_validation",
        "source_name": args.source_name,
        "source_url": safe_url,
        "source_url_sensitive_parts_redacted": redacted,
        "source_portal": SOURCE_PAGE,
        "acquisition_date": datetime.now(timezone.utc).date().isoformat(),
        "filename": destination.name,
        "file_size_bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Imported {destination.stat().st_size:,} bytes to {destination}")


if __name__ == "__main__":
    main()
