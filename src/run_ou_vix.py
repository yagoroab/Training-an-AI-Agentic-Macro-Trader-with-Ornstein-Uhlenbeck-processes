import numpy as np
import pandas as pd

from src.data.vix_loader import load_vix_csv
from src.data.market_loader import load_yahoo_adjclose
from src.models.ou_estimation import rolling_ou_params
from src.backtest.backtest_ou import backtest_ou
from src.plot_ou_results import plot_ou_results, plot_wealth_vs_vix


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.sqrt(252) * x.mean() / s)


def max_drawdown_additive(cum: np.ndarray) -> float:
    cum = np.asarray(cum, dtype=float)
    cum = cum[np.isfinite(cum)]
    if cum.size < 2:
        return 0.0
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())


def max_drawdown_wealth(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w)]
    if w.size < 2:
        return 0.0
    peak = np.maximum.accumulate(w)
    dd = w / peak - 1.0
    return float(dd.min())


def run_ou_vix(
    vix_path: str = "data/VIX_History.csv",
    traded_ticker: str = "VXX",
    traded_start: str = "2018-01-01",
    traded_end: str = "2026-02-07",
    dt: float = 1.0,
    window: int = 126,
    split_date: str = "2020-01-01",
    params: dict | None = None,
    make_plots: bool = True,
    print_report: bool = True,
) -> dict:
    """
    OU-based volatility strategy:
    - estimate OU on full VIX CSV history
    - trade VXX when available
    - use 2018-2019 in-sample and 2020+ out-of-sample
    """
    if params is None:
        params = dict(
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

            min_hold_days=7,
            stop_loss=0.12,
        )

    vix_full = load_vix_csv(vix_path).astype(float).sort_index()
    traded_px = load_yahoo_adjclose(
        traded_ticker,
        start=traded_start,
        end=traded_end,
    ).astype(float).sort_index()

    x_full = np.log(vix_full.values.astype(float))
    mu_full, kappa_full, sigma_full = rolling_ou_params(x_full, window=window, dt=dt)

    ou_full = pd.DataFrame(
        {
            "vix": vix_full.values.astype(float),
            "mu": mu_full,
            "kappa": kappa_full,
            "sigma": sigma_full,
        },
        index=vix_full.index,
    )

    trade_df = pd.concat(
        [
            ou_full,
            traded_px.rename("traded"),
        ],
        axis=1,
    ).dropna().sort_index()

    trade_df = trade_df.loc[trade_df.index >= pd.to_datetime(traded_start)].copy()

    if trade_df.empty:
        raise ValueError("No aligned tradable sample after merging VIX history with traded asset.")

    s = trade_df["vix"].astype(float)
    traded = trade_df["traded"].astype(float)

    x = np.log(s.values.astype(float))
    x_level = s.values.astype(float)
    traded_price = traded.values.astype(float)

    mu = trade_df["mu"].values.astype(float)
    kappa = trade_df["kappa"].values.astype(float)
    sigma = trade_df["sigma"].values.astype(float)

    test_mask = s.index >= split_date
    train_mask = ~test_mask
    test_dates = s.index[test_mask]

    res = backtest_ou(
        x=x,
        traded_price=traded_price,
        mu=mu,
        kappa=kappa,
        sigma=sigma,
        **params,
    )

    pnl = np.asarray(res["pnl"], dtype=float)
    pos = np.asarray(res["pos"], dtype=float)

    r = np.nan_to_num(pnl, nan=0.0)
    cum = np.cumsum(r)
    wealth = np.cumprod(1.0 + np.clip(r, -0.99, None))

    pnl_train = r[train_mask]
    pnl_test = r[test_mask]

    wealth_train = np.cumprod(1.0 + np.clip(pnl_train, -0.99, None))
    wealth_test = np.cumprod(1.0 + np.clip(pnl_test, -0.99, None))

    n_years_full = (s.index.max() - s.index.min()).days / 365.25
    final_wealth_full = float(wealth[-1])
    cagr_full = final_wealth_full ** (1.0 / n_years_full) - 1.0 if n_years_full > 0 else 0.0

    n_years_test = (test_dates.max() - test_dates.min()).days / 365.25 if test_dates.size else 0.0
    final_wealth_test = float(wealth_test[-1]) if wealth_test.size else 1.0
    cagr_test = final_wealth_test ** (1.0 / n_years_test) - 1.0 if n_years_test > 0 else 0.0

    metrics = {
        "data_points": int(len(x)),
        "start": str(s.index.min().date()),
        "end": str(s.index.max().date()),
        "window": float(window),
        "dt": float(dt),
        "split_date": split_date,
        "total_pnl_additive": float(cum[-1]),
        "sharpe_full": float(sharpe(r)),
        "max_dd_additive": float(max_drawdown_additive(cum)),
        "final_wealth_full": float(final_wealth_full),
        "cagr_full": float(cagr_full),
        "max_dd_wealth_full": float(max_drawdown_wealth(wealth)),
        "sharpe_train": float(sharpe(pnl_train)),
        "sharpe_test": float(sharpe(pnl_test)),
        "final_wealth_test": float(final_wealth_test),
        "cagr_test": float(cagr_test),
        "max_dd_wealth_test": float(max_drawdown_wealth(wealth_test)),
    }

    if print_report:
        print("=== OU VIX / VXX RUN ===")
        print(f"Signal source: full VIX CSV history | Traded: {traded_ticker}")
        print(f"Tradable data points: {len(x)} | Start: {s.index.min().date()} | End: {s.index.max().date()}")
        print(f"Window: {window} | dt: {dt}")

        print("\n--- Full sample ---")
        print(f"Total PnL (additive): {cum[-1]:.4f}")
        print(f"Sharpe: {sharpe(r):.3f}")
        print(f"Max Drawdown (additive): {max_drawdown_additive(cum):.4f}")

        print("\n--- Wealth process (full sample) ---")
        print(f"Final Wealth (start=1.0): {final_wealth_full:.4f}")
        print(f"CAGR: {cagr_full:.3%}")
        print(f"Wealth Max Drawdown: {max_drawdown_wealth(wealth):.3%}")

        print("\n--- Split performance (Sharpe) ---")
        print(f"Split date: {split_date}")
        print(f"Train Sharpe: {sharpe(pnl_train):.3f}")
        print(f"Test Sharpe: {sharpe(pnl_test):.3f}")

        print("\n--- Wealth process (TEST only) ---")
        print(f"Test Final Wealth (start=1.0): {final_wealth_test:.4f}")
        print(f"Test CAGR: {cagr_test:.3%}")
        print(f"Test Wealth Max Drawdown: {max_drawdown_wealth(wealth_test):.3%}")

        print("\n--- Exposure diagnostics ---")
        print(f"Average |position|: {np.mean(np.abs(pos)):.4f}")
        print(f"% time invested: {100*np.mean(np.abs(pos) > 0):.2f}%")
        print(f"Max |position|: {np.max(np.abs(pos)):.2f}")

        if "risky_weight" in res:
            risky_weight = np.asarray(res["risky_weight"], dtype=float)
            print(f"Average risky weight: {np.mean(risky_weight):.4f}")
            print(f"Max risky weight: {np.max(risky_weight):.4f}")
            print(f"% time meaningfully invested (>5% risky): {100*np.mean(risky_weight > 0.05):.2f}%")

        if "cash_weight" in res:
            cash_weight = np.asarray(res["cash_weight"], dtype=float)
            print(f"Average cash weight: {np.mean(cash_weight):.4f}")
            print(f"% time mostly in cash (>95% cash): {100*np.mean(cash_weight > 0.95):.2f}%")

        if "cash_yield" in res:
            cash_yield = np.asarray(res["cash_yield"], dtype=float)
            print(f"Average daily cash yield contribution: {np.mean(cash_yield):.6f}")

        dpos = np.diff(pos, prepend=pos[0])
        reb_th = float(params["rebalance_thresh"])
        print(f"Number of rebalances (>|{reb_th:.2f}|): {int(np.sum(np.abs(dpos) > reb_th))}")

        if "vol_mult" in res:
            vol_mult = np.asarray(res["vol_mult"], dtype=float)
            vol_mult = vol_mult[np.isfinite(vol_mult)]
            if vol_mult.size:
                print(f"Average vol multiplier: {np.mean(vol_mult):.4f}")
                print(f"Max vol multiplier: {np.max(vol_mult):.4f}")

        print("\n--- Return distribution ---")
        print(f"Mean daily return: {np.mean(r):.6f}")
        print(f"Std daily return: {np.std(r, ddof=1):.6f}")
        print(f"Min daily return: {np.min(r):.4f}")
        print(f"Max daily return: {np.max(r):.4f}")
        print(f"1% worst day: {np.percentile(r, 1):.4f}")
        print(f"5% worst day: {np.percentile(r, 5):.4f}")

        covid_mask = (s.index >= "2020-02-01") & (s.index <= "2020-06-30")
        if np.any(covid_mask):
            print("\n--- Stress window ---")
            print(f"COVID window total return: {np.sum(r[covid_mask]):.4f}")

        print("\n--- Wealth sanity ---")
        print(f"Final wealth: {wealth[-1]:.4f}")
        print(f"Check via cumprod: {np.cumprod(1 + r)[-1]:.4f}")

        print("\n--- Param sanity ---")
        print(f"kappa (min/med/max): {np.nanmin(kappa):.6f} / {np.nanmedian(kappa):.6f} / {np.nanmax(kappa):.6f}")
        print(f"sigma (min/med/max): {np.nanmin(sigma):.6f} / {np.nanmedian(sigma):.6f} / {np.nanmax(sigma):.6f}")

    if make_plots:
        plot_ou_results(
            dates=test_dates,
            x=x_level[test_mask],
            mu=np.exp(mu)[test_mask],
            z=np.asarray(res["z"], dtype=float)[test_mask],
            pnl=r[test_mask],
            wealth=wealth_test,
            z_entry=float(params["z_entry"]),
            z_exit=float(params["z_exit"]),
            risky_weight=np.asarray(res["risky_weight"], dtype=float)[test_mask],
            cash_weight=np.asarray(res["cash_weight"], dtype=float)[test_mask],
        )

        plot_wealth_vs_vix(
            dates=s.index,
            wealth=wealth,
            vix=x_level,
        )

        if print_report:
            print("Plots saved to /figures (test-only plots plus wealth_vs_vix)")

    return {
        "dates": s.index,
        "series": s,
        "traded_series": traded,
        "x_level": x_level,
        "x_log": x,
        "traded_price": traded_price,
        "mu": mu,
        "kappa": kappa,
        "sigma": sigma,
        "res": res,
        "pnl": r,
        "cum_additive": cum,
        "wealth_full": wealth,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "test_dates": test_dates,
        "wealth_test": wealth_test,
        "metrics": metrics,
        "params": params,
        "traded_ticker": traded_ticker,
    }


if __name__ == "__main__":
    run_ou_vix()







