import numpy as np
from src.data.vix_loader import load_vix_csv
from src.models.ou_estimation import rolling_ou_params
from src.backtest.backtest_ou import backtest_ou
from src.plot_ou_results import plot_ou_results


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
    dt: float = 1.0,
    window: int = 126,
    split_date: str = "2006-01-01",
    params: dict | None = None,
    make_plots: bool = True,
    print_report: bool = True,
) -> dict:
    """
    Runs EXACT OU VIX benchmark you already wrote.
    Returns a dict with arrays + metrics so other scripts (freezer, RL eval) can reuse it.
    """

    # ---- Load data ----
    s = load_vix_csv(vix_path)
    x_level = s.values.astype(float)
    x = np.log(x_level)

    # ---- Estimate OU params ----
    mu, kappa, sigma = rolling_ou_params(x, window=window, dt=dt)

    # ---- Train/Test split (for reporting/plotting) ----
    test_mask = s.index >= split_date
    train_mask = ~test_mask
    test_dates = s.index[test_mask]

    # ---- Strategy knobs (your exact defaults) ----
    if params is None:
        params = dict(
            cost_bps=1.0,
            carry_bps_per_day=0.2,   # IMPORTANT: keep tiny for benchmark
            exposure=0.20,

            # hysteresis thresholds (less churn, less always-in)
            z_entry=1.8,
            z_exit=0.9,
            z_cap=3.0,

            # leverage
            max_leverage=1.0,

            # vol targeting (less aggressive)
            vol_target=0.006,
            vol_window=20,
            max_vol_mult=2.0,
            vol_mult_floor=0.50,

            # regime filter
            kappa_min=0.01,
            hl_max=80.0,

            # turnover control
            pos_ema_alpha=1.0,
            rebalance_thresh=0.15,
        )

    # ---- Backtest (full sample; uses past-only info) ----
    res = backtest_ou(
        x=x,
        x_level=x_level,
        mu=mu,
        kappa=kappa,
        sigma=sigma,
        **params
    )

    pnl = np.asarray(res["pnl"], dtype=float)
    pos = np.asarray(res["pos"], dtype=float)

    r = np.nan_to_num(pnl, nan=0.0)
    cum = np.cumsum(r)
    wealth = np.cumprod(1.0 + np.clip(r, -0.99, None))

    # ---- Train/Test series ----
    pnl_train = r[train_mask]
    pnl_test = r[test_mask]

    wealth_train = np.cumprod(1.0 + np.clip(pnl_train, -0.99, None))
    wealth_test = np.cumprod(1.0 + np.clip(pnl_test, -0.99, None))

    # ---- Main performance ----
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
        print("=== OU VIX RUN ===")
        print(f"Data points: {len(x)} | Start: {s.index.min().date()} | End: {s.index.max().date()}")
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

        crisis_mask = (s.index >= "2008-01-01") & (s.index <= "2009-12-31")
        covid_mask = (s.index >= "2020-02-01") & (s.index <= "2020-06-30")

        print("\n--- Crisis windows ---")
        print(f"2008–2009 total return: {np.sum(r[crisis_mask]):.4f}")
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
        )
        if print_report:
            print("Plots saved to /figures (test-only plots)")

    # Return everything needed for freezing + RL comparison
    return {
        "dates": s.index,
        "series": s,
        "x_level": x_level,
        "x_log": x,
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
    }


if __name__ == "__main__":
    run_ou_vix()







