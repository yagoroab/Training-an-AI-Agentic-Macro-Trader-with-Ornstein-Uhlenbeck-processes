import os
import numpy as np
import matplotlib.pyplot as plt

def plot_ou_results(dates, x, mu, z, pnl):
    os.makedirs("figures", exist_ok=True)

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
    plt.axhline(1.5, linestyle="--", color="red", alpha=0.5)
    plt.axhline(-1.5, linestyle="--", color="red", alpha=0.5)
    plt.axhline(0.0, linestyle="--", color="gray", alpha=0.5)
    plt.title("OU Z-score")
    plt.tight_layout()
    plt.savefig("figures/ou_zscore.png")
    plt.close()

    # 3) Cumulative PnL
    cum_pnl = np.cumsum(pnl)
    plt.figure(figsize=(12, 4))
    plt.plot(dates, cum_pnl, label="Cumulative PnL")
    plt.title("OU Strategy Cumulative PnL")
    plt.tight_layout()
    plt.savefig("figures/ou_cumulative_pnl.png")
    plt.close()
