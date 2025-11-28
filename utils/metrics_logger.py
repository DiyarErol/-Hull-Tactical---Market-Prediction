import json
from pathlib import Path
from datetime import datetime


def save_metrics(metrics: dict, out_dir: str = "reports", name_prefix: str = "metrics"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"{name_prefix}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return str(out_path)
