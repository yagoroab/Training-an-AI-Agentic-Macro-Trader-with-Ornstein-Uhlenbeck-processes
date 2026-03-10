import os
import numpy as np
import matplotlib.pyplot as plt


def plot_ou_results(
    dates,
    x,
    mu,
    z,
    pnl,
    wealth=None,
    z_entry=1.5,
    z_exit=0.5,
    risky_weight=None,
    cash_weight=None,
):
    os.makedirs("figures", exist_ok=True)

    pnl = np.asarray(pnl, dtype=float)
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    x = np.asarray(x, dtype=float)

    if risky_weight is not None:
        risky_weight = np.asarray(risky_weight, dtype=float)

    if cash_weight is not None:
        cash_weight = np.asarray(cash_weight, dtype=float)

    # 1) VIX + OU mean
    plt.figure(figsize=(12, 5))
    plt.plot(dates, x, label="VIX", alpha=0.8)
    mu_plot = np.clip(mu, 0.0, np.nanpercentile(mu, 99.5))
    plt.plot(dates, mu_plot, label="OU mean (μₜ)", linewidth=2)
    plt.title("VIX and Rolling OU Mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/vix_ou_mean.png")
    plt.close()

    # 2) Z-score
    plt.figure(figsize=(12, 4))
    plt.plot(dates, z, label="OU z-score", color="black")
    plt.axhline(z_entry, linestyle="--", color="red", alpha=0.6, label="Entry threshold")
    plt.axhline(-z_entry, linestyle="--", color="red", alpha=0.6)
    plt.axhline(z_exit, linestyle="--", color="blue", alpha=0.5, label="Exit threshold")
    plt.axhline(-z_exit, linestyle="--", color="blue", alpha=0.5)
    plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
    plt.title("OU Z-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/ou_zscore.png")
    plt.close()

    # 3) Daily PnL
    plt.figure(figsize=(12, 4))
    plt.plot(dates, pnl, label="Daily PnL", color="black", linewidth=0.8)
    plt.axhline(0.0, linestyle="--", color="gray", alpha=0.7)
    plt.title("OU Strategy Daily PnL")
    plt.tight_layout()
    plt.savefig("figures/ou_daily_pnl.png")
    plt.close()

    # 4) PnL histogram
    pnl_clean = pnl[np.isfinite(pnl)]
    if pnl_clean.size > 10:
        m = pnl_clean.mean()
        s = pnl_clean.std(ddof=1)
        skew = float(np.mean(((pnl_clean - m) / (s + 1e-12)) ** 3)) if s > 0 else 0.0
    else:
        skew = 0.0

    plt.figure(figsize=(8, 4))
    plt.hist(pnl_clean, bins=80)
    plt.axvline(0.0, linestyle="--", color="gray", alpha=0.7)
    plt.title(f"Daily PnL Distribution (Skewness ≈ {skew:.2f})")
    plt.tight_layout()
    plt.savefig("figures/ou_pnl_hist.png")
    plt.close()

    # 5) Wealth curve
    if wealth is None:
        r = np.nan_to_num(pnl, nan=0.0)
        r = np.clip(r, -0.99, None)
        wealth = np.cumprod(1.0 + r)
    else:
        wealth = np.asarray(wealth, dtype=float)

    plt.figure(figsize=(12, 4))
    plt.plot(dates, wealth, label="Wealth (compounded)")
    plt.title("OU Strategy Wealth (Compounded)")
    plt.tight_layout()
    plt.savefig("figures/ou_wealth.png")
    plt.close()

    # 6) Wealth drawdown
    peak = np.maximum.accumulate(wealth)
    dd = (wealth / peak) - 1.0

    plt.figure(figsize=(12, 4))
    plt.plot(dates, dd, color="black")
    plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
    plt.title("OU Strategy Wealth Drawdown")
    plt.tight_layout()
    plt.savefig("figures/ou_wealth_drawdown.png")
    plt.close()

    # 7) Allocation over time
    if risky_weight is not None and cash_weight is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(dates, risky_weight, label="Risky allocation", linewidth=1.5)
        plt.plot(dates, cash_weight, label="Cash allocation", linewidth=1.5)
        plt.title("OU Strategy Allocation Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio weight")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/ou_allocation_over_time.png")
        plt.close()

        invested_share = float(np.mean(risky_weight > 0.05))
        cash_share = float(np.mean(cash_weight > 0.95))

        plt.figure(figsize=(6, 4))
        plt.bar(["Invested", "Mostly Cash"], [invested_share, cash_share])
        plt.title("Fraction of Time Invested vs Mostly in Cash")
        plt.ylabel("Fraction of observations")
        plt.tight_layout()
        plt.savefig("figures/ou_invested_vs_cash_bar.png")
        plt.close()

def plot_wealth_vs_vix(dates, wealth, vix, outpath="figures/wealth_vs_vix.png"):
    """
    Plot strategy wealth vs VIX level.
    Demonstrates how the strategy behaves relative to volatility regimes.
    """

    os.makedirs("figures", exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10,6))

    # Strategy wealth
    ax1.plot(dates, wealth, label="Strategy Wealth", color="blue", linewidth=2)
    ax1.set_ylabel("Strategy Wealth", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # VIX on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(dates, vix, label="VIX", color="red", alpha=0.6)
    ax2.set_ylabel("VIX Level", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Strategy Performance vs VIX Level")

    fig.tight_layout()
    plt.savefig(outpath)
    plt.close()
