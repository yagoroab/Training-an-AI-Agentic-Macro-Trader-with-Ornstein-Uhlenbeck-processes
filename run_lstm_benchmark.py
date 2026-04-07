from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.data.market_loader import load_yahoo_adjclose, prices_to_wealth
from src.data.vix_loader import load_vix_csv
from src.models.lstm_vxx import (
    backtest_lstm_probabilities,
    build_lstm_feature_frame,
    prepare_lstm_dataset,
    train_gru_classifier,
    train_lstm_classifier,
)
from src.plot_compare_results import (
    plot_drawdown_comparison,
    plot_wealth_comparison,
    print_annualized_table,
)
from src.run_ou_vix import run_ou_vix

RNN_DATA_START = "2018-01-01"
TRADED_START = "2020-05-01"
MODEL_SPLIT_DATE = "2022-05-01"
TRADED_END = "2025-07-31"


def _to_series(obj, name: str) -> pd.Series:
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"{name} must be a Series or 1-column DataFrame, got shape {obj.shape}")
        s = obj.iloc[:, 0].copy()
    else:
        raise TypeError(f"{name} must be a pandas Series or DataFrame, got {type(obj)}")

    s = s.astype(float)
    s.name = name
    return s.sort_index()


def _train_rnn_strategy(
    model_name: str,
    trainer,
    ds: dict,
    y_class: np.ndarray,
) -> tuple[pd.Series, dict, np.ndarray]:
    seed_probs = []
    for seed in [7, 42, 99]:
        model = trainer(
            x_train=ds["x"][ds["train_mask"]],
            y_train=y_class[ds["train_mask"]],
            seed=seed,
            epochs=20,
            batch_size=32,
        )
        seed_probs.append(model.predict(ds["x"], verbose=0).reshape(-1))

    probability_up = np.mean(np.vstack(seed_probs), axis=0)
    confidence_scale = float(np.std(probability_up[ds["train_mask"]] - 0.5))
    trade_cfg = {
        "LSTM Strategy": dict(max_leverage=0.22, probability_scale=1.50, deadband=0.30),
        "GRU Strategy": dict(max_leverage=0.50, probability_scale=1.75, deadband=0.20),
    }[model_name]
    bt = backtest_lstm_probabilities(
        dates=ds["dates"],
        traded_price=ds["traded_price"],
        probability_up=probability_up,
        cash_yield_annual=0.03,
        cost_bps=1.0,
        max_leverage=trade_cfg["max_leverage"],
        probability_scale=trade_cfg["probability_scale"],
        deadband=trade_cfg["deadband"],
        confidence_scale=confidence_scale,
        pos_ema_alpha=0.60,
        rebalance_thresh=0.02,
    )

    wealth = pd.Series(bt["wealth"], index=ds["dates"], name=model_name).astype(float)
    wealth = wealth / float(wealth.iloc[0])
    return wealth, bt, probability_up


def _build_dataset(
    feature_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    sequence_length: int,
    target_col: str,
    pos_threshold: float,
    neg_threshold: float,
) -> tuple[dict, np.ndarray]:
    ds = prepare_lstm_dataset(
        feature_df=feature_df,
        feature_cols=feature_cols,
        sequence_length=sequence_length,
        split_date=MODEL_SPLIT_DATE,
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


def main() -> None:
    os.makedirs("figures", exist_ok=True)

    vix = load_vix_csv("data/VIX_History.csv").astype(float).sort_index()
    vxx = load_yahoo_adjclose("VXX", start=RNN_DATA_START, end=TRADED_END).astype(float).sort_index()

    feature_df = build_lstm_feature_frame(vix=vix, traded=vxx)
    base_feature_cols = [
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
    lstm_feature_cols = [
        "log_vix_over_traded",
        "vix_ret_1",
        "traded_ret_1",
        "vix_ret_5",
        "traded_ret_5",
        "vix_vol_20",
        "traded_vol_20",
    ]
    gru_feature_cols = base_feature_cols

    lstm_ds, lstm_y_class = _build_dataset(
        feature_df,
        feature_cols=lstm_feature_cols,
        sequence_length=40,
        target_col="target_ret_3d",
        pos_threshold=0.0075,
        neg_threshold=-0.0075,
    )
    gru_ds, gru_y_class = _build_dataset(
        feature_df,
        feature_cols=gru_feature_cols,
        sequence_length=55,
        target_col="target_ret_5d",
        pos_threshold=0.010,
        neg_threshold=-0.010,
    )

    lstm_wealth, lstm_bt, lstm_prob = _train_rnn_strategy(
        "LSTM Strategy",
        train_lstm_classifier,
        lstm_ds,
        lstm_y_class,
    )
    gru_wealth, gru_bt, gru_prob = _train_rnn_strategy(
        "GRU Strategy",
        train_gru_classifier,
        gru_ds,
        gru_y_class,
    )

    lstm_test_mask = lstm_wealth.index >= pd.to_datetime(MODEL_SPLIT_DATE)
    lstm_wealth = lstm_wealth.loc[lstm_test_mask]
    lstm_wealth = lstm_wealth / float(lstm_wealth.iloc[0])
    lstm_pos = np.asarray(lstm_bt["pos"], dtype=float)[lstm_test_mask]
    lstm_risky = np.asarray(lstm_bt["risky_weight"], dtype=float)[lstm_test_mask]
    lstm_prob_test = np.asarray(lstm_prob, dtype=float)[lstm_test_mask]

    gru_test_mask = gru_wealth.index >= pd.to_datetime(MODEL_SPLIT_DATE)
    gru_wealth = gru_wealth.loc[gru_test_mask]
    gru_wealth = gru_wealth / float(gru_wealth.iloc[0])
    gru_pos = np.asarray(gru_bt["pos"], dtype=float)[gru_test_mask]
    gru_risky = np.asarray(gru_bt["risky_weight"], dtype=float)[gru_test_mask]
    gru_prob_test = np.asarray(gru_prob, dtype=float)[gru_test_mask]

    ou_out = run_ou_vix(
        traded_start=TRADED_START,
        traded_end=TRADED_END,
        split_date=MODEL_SPLIT_DATE,
        make_plots=False,
        print_report=False,
    )
    ou_wealth = pd.Series(
        ou_out["wealth_test"],
        index=pd.to_datetime(ou_out["test_dates"]),
        name="OU Strategy",
    ).astype(float)
    ou_wealth = ou_wealth / float(ou_wealth.iloc[0])

    start = str(lstm_wealth.index.min().date())
    end = str(lstm_wealth.index.max().date())
    sp500_px = load_yahoo_adjclose("^GSPC", start=start, end=end)
    sp500_wealth = prices_to_wealth(_to_series(sp500_px, "S&P 500 (^GSPC)"))
    sp500_wealth = _to_series(sp500_wealth, "S&P 500 (^GSPC)")
    sp500_wealth = sp500_wealth / float(sp500_wealth.iloc[0])

    benchmarks = {
        "GRU Strategy": gru_wealth,
        "OU Strategy": ou_wealth,
        "S&P 500 (^GSPC)": sp500_wealth,
    }

    plot_wealth_comparison(
        ou_wealth=lstm_wealth,
        benchmarks_wealth=benchmarks,
        outpath="figures/rnn_ou_sp500_wealth.png",
        title="Wealth Comparison: LSTM vs GRU vs OU Strategy vs S&P 500",
        log_scale=False,
    )

    plot_wealth_comparison(
        ou_wealth=lstm_wealth,
        benchmarks_wealth=benchmarks,
        outpath="figures/rnn_ou_sp500_wealth_log.png",
        title="Wealth Comparison (Log Scale): LSTM vs GRU vs OU Strategy vs S&P 500",
        log_scale=True,
    )

    plot_drawdown_comparison(
        ou_wealth=lstm_wealth,
        benchmarks_wealth=benchmarks,
        outpath="figures/rnn_ou_sp500_drawdown.png",
        title="Drawdown Comparison: LSTM vs GRU vs OU Strategy vs S&P 500",
    )
    
    

    print("\n=== LSTM Strategy Diagnostics ===")
    print(f"Sample: {lstm_wealth.index.min().date()} -> {lstm_wealth.index.max().date()}")
    print(f"Final Wealth: {lstm_wealth.iloc[-1]:.4f}")
    print(f"Average P(up): {np.mean(lstm_prob_test):.4f}")
    print(f"Average |position|: {np.mean(np.abs(lstm_pos)):.4f}")
    print(f"% time invested: {100*np.mean(np.abs(lstm_pos) > 0):.2f}%")
    print(f"Average risky weight: {np.mean(lstm_risky):.4f}")

    print("\n=== GRU Strategy Diagnostics ===")
    print(f"Sample: {gru_wealth.index.min().date()} -> {gru_wealth.index.max().date()}")
    print(f"Final Wealth: {gru_wealth.iloc[-1]:.4f}")
    print(f"Average P(up): {np.mean(gru_prob_test):.4f}")
    print(f"Average |position|: {np.mean(np.abs(gru_pos)):.4f}")
    print(f"% time invested: {100*np.mean(np.abs(gru_pos) > 0):.2f}%")
    print(f"Average risky weight: {np.mean(gru_risky):.4f}")

    print_annualized_table(
        ou_wealth=lstm_wealth,
        benchmarks_wealth=benchmarks,
    )

    print("\nSaved figures:")
    print("- figures/rnn_ou_sp500_wealth.png")
    print("- figures/rnn_ou_sp500_wealth_log.png")
    print("- figures/rnn_ou_sp500_drawdown.png")


if __name__ == "__main__":
    main()
