import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import DELAY_METRICS_PATH  # noqa: E402
from app.delay_training import load_delay_model  # noqa: E402


if __name__ == "__main__":
    bundle = load_delay_model()
    if bundle["status"] != "available":
        print(f"Shipment-delay model unavailable: {bundle['reason']}")
        raise SystemExit(1)
    with DELAY_METRICS_PATH.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    test = metrics["test"]
    print(f"Model: {metrics['model']}")
    print(f"Dataset: {metrics['dataset_identity']['source_name']}")
    print(f"Final chronological test samples: {test['sample_count']}")
    print(
        "Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | "
        "ROC-AUC: {roc_auc:.3f}".format(
            precision=test["precision"],
            recall=test["recall"],
            f1=test["f1"],
            roc_auc=test["roc_auc"],
        )
    )
    print(f"Evaluation: {DELAY_METRICS_PATH}")
