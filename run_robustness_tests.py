from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.market_loader import load_yahoo_adjclose
from src.data.vix_loader import load_vix_csv
from src.models.lstm_vxx import (
    backtest_lstm_probabilities,
    build_lstm_feature_frame,
    prepare_lstm_dataset,
    train_gru_classifier,
    train_lstm_classifier,
)
from src.run_ou_vix import run_ou_vix


CORE_FILES = [
    Path("src/backtest/backtest_ou.py"),
    Path("src/run_ou_vix.py"),
    Path("src/models/lstm_vxx.py"),
    Path("run_lstm_benchmark.py"),
]

RNN_DATA_START = "2018-01-01"
RNN_TRADED_START = "2020-05-01"
RNN_TRADED_END = "2026-02-07"
RNN_SPLIT_DATE = "2022-05-01"


def _build_rnn_case_dataset(feature_df: pd.DataFrame, *, trainer, split_date: str) -> tuple[dict, np.ndarray]:
    if trainer is train_lstm_classifier:
        feature_cols = [
            "log_vix_over_traded",
            "vix_ret_1",
            "traded_ret_1",
            "vix_ret_5",
            "traded_ret_5",
            "vix_vol_20",
            "traded_vol_20",
        ]
        sequence_length = 40
        target_col = "target_ret_3d"
        pos_threshold = 0.0075
        neg_threshold = -0.0075
    else:
        feature_cols = [
            "log_vix",
            "log_traded",
            "log_vix_over_traded",
            "vix_ret_1",
            "traded_ret_1",
            "vix_ret_5",
            "traded_ret_5",
            "vix_ret_10",
            "traded_ret_10",
            "vix_ret_20",
            "traded_ret_20",
            "vix_vol_20",
            "traded_vol_20",
        ]
        sequence_length = 55
        target_col = "target_ret_5d"
        pos_threshold = 0.010
        neg_threshold = -0.010

    ds = prepare_lstm_dataset(
        feature_df=feature_df,
        feature_cols=feature_cols,
        sequence_length=sequence_length,
        split_date=split_date,
        target_col=target_col,
    )
    y_class = np.where(ds["y"] > pos_threshold, 1.0, np.where(ds["y"] < neg_threshold, 0.0, np.nan))
    valid_mask = np.isfinite(y_class)
    ds = {
        **ds,
        "x": ds["x"][valid_mask],
        "y": ds["y"][valid_mask],
        "dates": ds["dates"][valid_mask],
        "traded_price": ds["traded_price"][valid_mask],
        "train_mask": ds["train_mask"][valid_mask],
    }
    return ds, y_class[valid_mask].astype(np.float32)


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
        "maxdd": float(metrics["max_dd_non_compounded_full"]),
        "wealth": float(metrics["final_wealth_full"]),
        "test_cagr": float(metrics["cagr_test"]),
        "test_sharpe": float(metrics["sharpe_test"]),
        "test_maxdd": float(metrics["max_dd_non_compounded_test"]),
        "test_wealth": float(metrics["final_wealth_test"]),
    }


def _cagr_from_wealth(wealth: np.ndarray, dates: pd.DatetimeIndex) -> float:
    wealth = np.asarray(wealth, dtype=float)
    if wealth.size < 2:
        return 0.0
    years = (dates[-1] - dates[0]).days / 365.25
    final = float(wealth[-1])
    return float(final ** (1.0 / years) - 1.0) if years > 0 and final > 0 else 0.0


def _sharpe_from_pnl(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size < 2:
        return 0.0
    std = pnl.std(ddof=1)
    if std <= 0:
        return 0.0
    return float(np.sqrt(252.0) * pnl.mean() / std)


def _max_dd_non_compounded(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size < 2:
        return 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def run_rnn_case(
    label: str,
    *,
    trainer,
    seed: int = 42,
    split_date: str = RNN_SPLIT_DATE,
    cost_bps: float = 1.0,
    ensemble: bool = False,
) -> dict:
    vix = load_vix_csv("data/VIX_History.csv").astype(float).sort_index()
    vxx = load_yahoo_adjclose("VXX", start=RNN_DATA_START, end=RNN_TRADED_END).astype(float).sort_index()

    feature_df = build_lstm_feature_frame(vix=vix, traded=vxx)
    ds, y_class = _build_rnn_case_dataset(feature_df, trainer=trainer, split_date=split_date)

    seeds = [7, 42, 99] if ensemble else [seed]
    prob_up_list = []
    for fit_seed in seeds:
        model = trainer(
            x_train=ds["x"][ds["train_mask"]],
            y_train=y_class[ds["train_mask"]],
            seed=fit_seed,
            epochs=20,
            batch_size=32,
        )
        prob_up_list.append(model.predict(ds["x"], verbose=0).reshape(-1))
    prob_up = np.mean(np.vstack(prob_up_list), axis=0)
    confidence_scale = float(np.std(prob_up[ds["train_mask"]] - 0.5))
    trade_cfg = (
        dict(max_leverage=0.22, probability_scale=1.50, deadband=0.30)
        if trainer is train_lstm_classifier
        else dict(max_leverage=0.50, probability_scale=1.75, deadband=0.20)
    )

    bt = backtest_lstm_probabilities(
        dates=ds["dates"],
        traded_price=ds["traded_price"],
        probability_up=prob_up,
        cash_yield_annual=0.03,
        cost_bps=cost_bps,
        max_leverage=trade_cfg["max_leverage"],
        probability_scale=trade_cfg["probability_scale"],
        deadband=trade_cfg["deadband"],
        confidence_scale=confidence_scale,
        pos_ema_alpha=0.60,
        rebalance_thresh=0.02,
    )

    test_mask = ds["dates"] >= pd.to_datetime(split_date)
    test_dates = ds["dates"][test_mask]
    wealth_test = np.asarray(bt["wealth"], dtype=float)[test_mask]
    wealth_test = wealth_test / float(wealth_test[0]) if wealth_test.size else np.array([1.0])
    pnl_test = np.asarray(bt["pnl"], dtype=float)[test_mask]

    return {
        "label": label,
        "split_date": split_date,
        "seed": seed,
        "cost_bps": cost_bps,
        "test_cagr": _cagr_from_wealth(wealth_test, test_dates),
        "test_sharpe": _sharpe_from_pnl(pnl_test),
        "test_maxdd": _max_dd_non_compounded(pnl_test),
        "test_wealth": float(wealth_test[-1]) if wealth_test.size else 1.0,
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

    lstm_seed_results = []
    gru_seed_results = []
    for seed in [7, 42, 99]:
        lstm_seed_results.append(run_rnn_case(f"LSTM_SEED_{seed}", trainer=train_lstm_classifier, seed=seed))
        gru_seed_results.append(run_rnn_case(f"GRU_SEED_{seed}", trainer=train_gru_classifier, seed=seed))
    print_table("LSTM SEED ROBUSTNESS", "MODEL", lstm_seed_results, test_metrics=True)
    print_table("GRU SEED ROBUSTNESS", "MODEL", gru_seed_results, test_metrics=True)

    rnn_ensemble_results = [
        run_rnn_case("LSTM_ENSEMBLE", trainer=train_lstm_classifier, ensemble=True),
        run_rnn_case("GRU_ENSEMBLE", trainer=train_gru_classifier, ensemble=True),
    ]
    print_table("RNN ENSEMBLE TEST", "MODEL", rnn_ensemble_results, test_metrics=True)

    rnn_late_split_results = [
        run_rnn_case("LSTM_SPLIT_2023", trainer=train_lstm_classifier, split_date="2023-05-01", ensemble=True),
        run_rnn_case("GRU_SPLIT_2023", trainer=train_gru_classifier, split_date="2023-05-01", ensemble=True),
    ]
    print_table("RNN LATER SPLIT TEST", "MODEL", rnn_late_split_results, test_metrics=True)

    rnn_cost_results = [
        run_rnn_case("LSTM_COST_1", trainer=train_lstm_classifier, cost_bps=1.0, ensemble=True),
        run_rnn_case("LSTM_COST_5", trainer=train_lstm_classifier, cost_bps=5.0, ensemble=True),
        run_rnn_case("LSTM_COST_10", trainer=train_lstm_classifier, cost_bps=10.0, ensemble=True),
        run_rnn_case("GRU_COST_1", trainer=train_gru_classifier, cost_bps=1.0, ensemble=True),
        run_rnn_case("GRU_COST_5", trainer=train_gru_classifier, cost_bps=5.0, ensemble=True),
        run_rnn_case("GRU_COST_10", trainer=train_gru_classifier, cost_bps=10.0, ensemble=True),
    ]
    print_table("RNN TRANSACTION COST TEST", "MODEL", rnn_cost_results, test_metrics=True)

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
