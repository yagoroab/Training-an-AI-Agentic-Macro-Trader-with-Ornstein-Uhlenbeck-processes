# run_freeze_baseline.py
# Freezes the baseline EXACTLY as produced by run_ou_vix.py (no new strategy code)

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.run_ou_vix import run_ou_vix

def main():
    out_dir = Path("artifacts/baseline_from_run_ou_vix")
    out_dir.mkdir(parents=True, exist_ok=True)

    out = run_ou_vix(make_plots=False, print_report=False)

    pnl = np.asarray(out["pnl"], dtype=float)
    wealth = np.asarray(out["wealth_full"], dtype=float)
    metrics = dict(out["metrics"])  # copy

    # add counts
    train_mask = np.asarray(out["train_mask"])
    test_mask = np.asarray(out["test_mask"])
    metrics["n_obs"] = int(pnl.size)
    metrics["n_train"] = int(train_mask.sum())
    metrics["n_test"] = int(test_mask.sum())

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save curve (full sample)
    dates = out["dates"]
    curve = pd.DataFrame({
        "pnl": pnl,
        "wealth": wealth,
        "pos": np.asarray(out["res"].get("pos"), dtype=float),
        "z": np.asarray(out["res"].get("z"), dtype=float),
    }, index=dates)
    curve.to_csv(out_dir / "equity_curve.csv", index=True)

    # Also save the exact param dict used (so baseline is fully specified)
    with open(out_dir / "strategy_params.json", "w") as f:
        json.dump(out["params"], f, indent=2)

    print(f"Frozen baseline saved to: {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
