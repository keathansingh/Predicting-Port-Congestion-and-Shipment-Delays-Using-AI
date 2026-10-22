import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import METRICS_PATH  # noqa: E402


if __name__ == "__main__":
    if not METRICS_PATH.is_file():
        raise SystemExit("Evaluation artifact is missing. Run scripts\\train_model.py first.")
    print(json.dumps(json.loads(METRICS_PATH.read_text(encoding="utf-8")), indent=2))
