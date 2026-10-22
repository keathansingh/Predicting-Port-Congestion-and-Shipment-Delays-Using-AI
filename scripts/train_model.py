import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.model_training import train_and_evaluate  # noqa: E402


if __name__ == "__main__":
    result = train_and_evaluate()
    print(f"Selected model: {result['selected_model']}")
    print(f"Test samples: {result['test']['sample_count']}")
    print(f"Metrics: {ROOT / 'artifacts' / 'metrics' / 'evaluation.json'}")
