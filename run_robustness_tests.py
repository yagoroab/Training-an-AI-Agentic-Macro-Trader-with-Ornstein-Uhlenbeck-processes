from __future__ import annotations

import hashlib
from pathlib import Path

from src.run_ou_vix import run_ou_vix


CORE_FILES = [
    Path("src/backtest/backtest_ou.py"),
    Path("src/run_ou_vix.py"),
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_hashes(paths: list[Path]) -> dict[Path, str]:
    return {path: file_sha256(path) for path in paths}


def run_case(label: str, *, split_date: str = "2020-01-01", params_override: dict | None = None) -> dict:
    params_override = params_override or {}
    out = run_ou_vix(
        split_date=split_date,
        params=params_override if params_override else None,
        make_plots=False,
        print_report=False,
    )
    metrics = out["metrics"]
    return {
        "label": label,
        "split_date": split_date,
        "cagr": float(metrics["cagr_full"]),
        "sharpe": float(metrics["sharpe_full"]),
        "maxdd": float(metrics["max_dd_wealth_full"]),
        "wealth": float(metrics["final_wealth_full"]),
        "test_cagr": float(metrics["cagr_test"]),
        "test_sharpe": float(metrics["sharpe_test"]),
        "test_maxdd": float(metrics["max_dd_wealth_test"]),
        "test_wealth": float(metrics["final_wealth_test"]),
    }


def print_metric_block(title: str, result: dict) -> None:
    print(title)
    print(f"Sharpe: {result['sharpe']:.3f}")
    print(f"CAGR: {result['cagr']:.3%}")
    print(f"Max drawdown: {result['maxdd']:.3%}")
    print(f"Final wealth: {result['wealth']:.4f}")
    print()


def print_table(title: str, label_header: str, results: list[dict], *, test_metrics: bool = False) -> None:
    print(title)
    print("---------------------------------------------------")
    if test_metrics:
        print(f"{label_header:<18} {'TEST CAGR':>10} {'TEST SHARPE':>12} {'TEST MAXDD':>11} {'TEST WEALTH':>12}")
    else:
        print(f"{label_header:<18} {'CAGR':>8} {'SHARPE':>8} {'MAXDD':>8} {'WEALTH':>10}")
    print("---------------------------------------------------")
    for result in results:
        if test_metrics:
            print(
                f"{result['label']:<18} "
                f"{result['test_cagr']:>9.2%} "
                f"{result['test_sharpe']:>12.3f} "
                f"{result['test_maxdd']:>10.2%} "
                f"{result['test_wealth']:>12.4f}"
            )
        else:
            print(
                f"{result['label']:<18} "
                f"{result['cagr']:>7.2%} "
                f"{result['sharpe']:>8.3f} "
                f"{result['maxdd']:>7.2%} "
                f"{result['wealth']:>10.4f}"
            )
    print("---------------------------------------------------")
    print()


def main() -> None:
    start_hashes = snapshot_hashes(CORE_FILES)
    start_files = {path.resolve() for path in Path(".").rglob("*") if path.is_file()}

    baseline = run_case("BASELINE")
    print_metric_block("BASELINE RESULTS", baseline)

    parameter_robustness = []
    parameter_matrix = [
        ("HOLD_5", {"min_hold_days": 5}),
        ("HOLD_10", {"min_hold_days": 10}),
        ("STOP_10", {"stop_loss": 0.10}),
        ("STOP_15", {"stop_loss": 0.15}),
        ("NO_HOLD_RULE", {"min_hold_days": 0}),
    ]
    for label, params_override in parameter_matrix:
        parameter_robustness.append(run_case(label, params_override=params_override))
    print_table("PARAMETER ROBUSTNESS", "TEST NAME", parameter_robustness)

    transaction_cost_results = []
    transaction_cost_matrix = [
        ("COST_1", {"cost_bps": 1.0}),
        ("COST_5", {"cost_bps": 5.0}),
        ("COST_10", {"cost_bps": 10.0}),
        ("COST_25", {"cost_bps": 25.0}),
    ]
    for label, params_override in transaction_cost_matrix:
        transaction_cost_results.append(run_case(label, params_override=params_override))
    print_table("TRANSACTION COST STRESS TEST", "COST TEST", transaction_cost_results)

    walk_forward_results = []
    walk_forward_splits = [
        ("SPLIT_2020", "2020-01-01"),
        ("SPLIT_2021", "2021-01-01"),
        ("SPLIT_2022", "2022-01-01"),
        ("SPLIT_2023", "2023-01-01"),
        ("SPLIT_2024", "2024-01-01"),
    ]
    for label, split_date in walk_forward_splits:
        walk_forward_results.append(run_case(label, split_date=split_date))
    print_table("WALK FORWARD TEST", "SPLIT YEAR", walk_forward_results, test_metrics=True)

    end_hashes = snapshot_hashes(CORE_FILES)
    if start_hashes != end_hashes:
        raise RuntimeError("Core files changed during robustness tests.")

    end_files = {path.resolve() for path in Path(".").rglob("*") if path.is_file()}
    created_files = sorted(path for path in (end_files - start_files) if path.name != "run_robustness_tests.py")
    if created_files:
        raise RuntimeError(
            "Unexpected files were created during robustness tests:\n"
            + "\n".join(str(path) for path in created_files)
        )

    print("BASE CODE RESTORED")
    print("ROBUSTNESS TEST SCRIPT CREATED")


if __name__ == "__main__":
    main()
