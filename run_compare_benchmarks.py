from __future__ import annotations

import pandas as pd

from src.run_ou_vix import run_ou_vix
from src.data.market_loader import load_yahoo_adjclose, prices_to_wealth
from src.plot_compare_results import (
    plot_wealth_comparison,
    plot_drawdown_comparison,
    plot_excess_vs_benchmark,
    print_annualized_table,
)


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


def main():
    USE_TEST_ONLY = False

    ou_params = dict(
        cost_bps=1.0,
        carry_bps_per_day=0.2,
        cash_yield_annual=0.02,

        exposure=1.00,

        z_entry=1.25,
        z_exit=0.35,
        z_cap=3.0,

        max_leverage=1.25,
        vxx_short_bias=0.30,

        vol_target=0.012,
        vol_window=20,
        max_vol_mult=1.75,
        vol_mult_floor=0.60,

        kappa_min=0.01,
        hl_max=80.0,

        pos_ema_alpha=1.0,
        rebalance_thresh=0.10,
    )

    ou_out = run_ou_vix(
        vix_path="data/VIX_History.csv",
        traded_ticker="VXX",
        traded_start="2018-01-01",
        traded_end="2026-02-07",
        split_date="2020-01-01",
        params=ou_params,
        make_plots=False,
        print_report=False,
    )

    if USE_TEST_ONLY:
        dates = pd.to_datetime(ou_out["test_dates"])
        ou_wealth_arr = ou_out["wealth_test"]
        tag = "test"
    else:
        dates = pd.to_datetime(ou_out["dates"])
        ou_wealth_arr = ou_out["wealth_full"]
        tag = "full"

    ou_wealth = pd.Series(ou_wealth_arr, index=dates, name="OU Strategy").astype(float).sort_index()
    ou_wealth = ou_wealth / float(ou_wealth.iloc[0])

    start = str(ou_wealth.index.min().date())
    end = str(ou_wealth.index.max().date())
    print(f"Master date range: {start} -> {end} ({tag})")

    benchmarks = {
        "VXX Buy & Hold": "VXX",
        "S&P 500 (^GSPC)": "^GSPC",
    }

    bench_wealth = {}
    for label, ticker in benchmarks.items():
        px = load_yahoo_adjclose(ticker, start=start, end=end)
        px = _to_series(px, name=label)
        w = prices_to_wealth(px)
        w = _to_series(w, name=label)
        w = w / float(w.iloc[0])
        bench_wealth[label] = w

    plot_wealth_comparison(
        ou_wealth=ou_wealth,
        benchmarks_wealth=bench_wealth,
        outpath=f"figures/ou_vs_benchmarks_wealth_{tag}.png",
        title="Wealth Comparison: OU Strategy vs VXX Buy & Hold and S&P 500",
        log_scale=False,
    )

    plot_wealth_comparison(
        ou_wealth=ou_wealth,
        benchmarks_wealth=bench_wealth,
        outpath=f"figures/ou_vs_benchmarks_wealth_{tag}_log.png",
        title="Wealth Comparison (Log Scale): OU Strategy vs VXX Buy & Hold and S&P 500",
        log_scale=True,
    )

    plot_wealth_comparison(
        ou_wealth=ou_wealth,
        benchmarks_wealth={"VXX Buy & Hold": bench_wealth["VXX Buy & Hold"]},
        outpath=f"figures/ou_vs_vxx_wealth_{tag}.png",
        title="Wealth Comparison: OU Strategy vs VXX Buy & Hold",
        log_scale=False,
    )

    plot_wealth_comparison(
        ou_wealth=ou_wealth,
        benchmarks_wealth={"S&P 500 (^GSPC)": bench_wealth["S&P 500 (^GSPC)"]},
        outpath=f"figures/ou_vs_sp500_wealth_{tag}.png",
        title="Wealth Comparison: OU Strategy vs S&P 500",
        log_scale=False,
    )

    plot_drawdown_comparison(
        ou_wealth=ou_wealth,
        benchmarks_wealth=bench_wealth,
        outpath=f"figures/ou_vs_benchmarks_drawdown_{tag}.png",
        title="Drawdown Comparison: OU Strategy vs VXX Buy & Hold and S&P 500",
    )

    plot_excess_vs_benchmark(
        ou_wealth=ou_wealth,
        benchmark_wealth=bench_wealth["VXX Buy & Hold"],
        outpath_ratio=f"figures/ou_over_vxx_ratio_{tag}.png",
        outpath_log_ratio=f"figures/ou_over_vxx_logratio_{tag}.png",
        benchmark_name="VXX Buy & Hold",
    )

    print_annualized_table(
        ou_wealth=ou_wealth,
        benchmarks_wealth=bench_wealth,
    )

    print("\nSaved figures:")
    print(f"- figures/ou_vs_benchmarks_wealth_{tag}.png")
    print(f"- figures/ou_vs_benchmarks_wealth_{tag}_log.png")
    print(f"- figures/ou_vs_vxx_wealth_{tag}.png")
    print(f"- figures/ou_vs_sp500_wealth_{tag}.png")
    print(f"- figures/ou_vs_benchmarks_drawdown_{tag}.png")
    print(f"- figures/ou_over_vxx_ratio_{tag}.png")
    print(f"- figures/ou_over_vxx_logratio_{tag}.png")


if __name__ == "__main__":
    main()