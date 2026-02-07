import numpy as np
from src.data.vix_loader import load_vix_csv
from src.models.ou_estimation import rolling_ou_params
from src.backtest.backtest_ou import backtest_ou
from src.plot_ou_results import plot_ou_results


def sharpe(pnl: np.ndarray) -> float:
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size < 2:
        return 0.0
    s = pnl.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.sqrt(252) * pnl.mean() / s)

def max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())

if __name__ == "__main__":
    # 1) Put your VIX CSV inside repo_root/data/
    # 2) Set the filename here:
    VIX_PATH = "data/VIX_History.csv"
 # <-- change if your file has a different name

    s = load_vix_csv(VIX_PATH)
    x_level = s.values.astype(float)
    x = np.log(x_level)

    dt = 1.0      # daily
    window = 252  # ~1 trading year

    mu, kappa, sigma = rolling_ou_params(x, window=window, dt=dt)
    res = backtest_ou(x, mu, kappa, sigma, cost_bps=1.0, z_entry=1.5, z_exit=0.3)

    pnl = res["pnl"]
    cum = np.cumsum(pnl)

    print("=== OU VIX RUN ===")
    print(f"Data points: {len(x)} | Start: {s.index.min().date()} | End: {s.index.max().date()}")
    print(f"Window: {window} | dt: {dt}")
    print(f"Total PnL: {cum[-1]:.4f}")
    print(f"Sharpe: {sharpe(pnl):.3f}")
    print(f"Max Drawdown: {max_drawdown(cum):.4f}")

    print("--- Param sanity ---")
    print(f"kappa (min/med/max): {np.nanmin(kappa):.6f} / {np.nanmedian(kappa):.6f} / {np.nanmax(kappa):.6f}")
    print(f"sigma (min/med/max): {np.nanmin(sigma):.6f} / {np.nanmedian(sigma):.6f} / {np.nanmax(sigma):.6f}")
    plot_ou_results(
        dates=s.index,
        x=x_level,
        mu=np.exp(mu),
        z=res["z"],
        pnl=pnl
)

print("Plots saved to /figures")

