import json
from pathlib import Path
from datetime import datetime
import numpy as np


def save_metrics(metrics: dict, out_dir: str = "reports", name_prefix: str = "metrics"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"{name_prefix}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return str(out_path)


def compute_fin_metrics(returns: np.ndarray, tc_bps: float = 2.0):
    """Compute Sharpe and Max Drawdown on daily returns with transaction costs (bps)."""
    tc = tc_bps / 10000.0
    net = returns - tc
    mu = np.mean(net)
    sigma = np.std(net) + 1e-12
    sharpe = (mu / sigma) * np.sqrt(252)
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / (peak + 1e-12)
    max_dd = float(np.min(drawdown))
    return {"sharpe": float(sharpe), "max_drawdown": max_dd}
