import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    signal_label="Signal",
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

    # 1) Signal + OU mean
    plt.figure(figsize=(12, 5))
    plt.plot(dates, x, label=signal_label, alpha=0.8)
    mu_plot = np.clip(mu, 0.0, np.nanpercentile(mu, 99.5))
    plt.plot(dates, mu_plot, label="OU mean (μₜ)", linewidth=2)
    plt.title(f"{signal_label} and Rolling OU Mean")
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

    # 4) Yearly PnL by tradable year
    pnl_mask = np.isfinite(pnl)
    pnl_dates = pd.to_datetime(np.asarray(dates)[pnl_mask])
    pnl_clean = pnl[pnl_mask]

    if pnl_clean.size:
        yearly_pnl = (
            pd.DataFrame({"date": pnl_dates, "pnl": pnl_clean})
            .assign(year=lambda df: df["date"].dt.year)
            .groupby("year", sort=True)["pnl"]
            .sum()
        )

        years = yearly_pnl.index.astype(str).tolist()
        values = yearly_pnl.to_numpy(dtype=float)
        colors = ["#2e8b57" if value >= 0.0 else "#c0392b" for value in values]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(years, values, color=colors, alpha=0.9)
        plt.axhline(0.0, linestyle="--", color="gray", alpha=0.7)
        plt.title("Yearly PnL by Tradable Year")
        plt.xlabel("Year")
        plt.ylabel("Total PnL")

        value_span = float(np.max(np.abs(values))) if values.size else 0.0
        offset = max(value_span * 0.03, 0.002)

        for bar, value in zip(bars, values):
            x_pos = bar.get_x() + bar.get_width() / 2.0
            if value >= 0.0:
                y_pos = value + offset
                va = "bottom"
            else:
                y_pos = value - offset
                va = "top"
            plt.text(x_pos, y_pos, f"{value:.3f}", ha="center", va=va, fontsize=9)

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

    # 6) Non-compounded drawdown from cumulative PnL
    cum_pnl = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak

    plt.figure(figsize=(12, 4))
    plt.plot(dates, dd, color="black")
    plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
    plt.title("OU Strategy Drawdown (Non-Compounded)")
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
