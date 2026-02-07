import numpy as np
from src.strategies.ou_threshold import ou_zscore, target_position_from_z

def backtest_ou(x, mu, kappa, sigma, cost_bps=1.0, z_entry=1.5, z_exit=0.3):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    cost = cost_bps * 1e-4
    pos = 0

    pnl = np.zeros(len(x), dtype=float)
    z_arr = np.full(len(x), np.nan)

    for t in range(1, len(x)):
        if np.isnan(mu[t]) or np.isnan(kappa[t]) or np.isnan(sigma[t]):
            continue

        z = ou_zscore(x[t], mu[t], kappa[t], sigma[t])
        z_arr[t] = z

        new_pos = target_position_from_z(z, z_entry=z_entry, z_exit=z_exit, prev_pos=pos)

        dx = x[t] - x[t-1]
        trade_cost = cost * abs(new_pos - pos)
        pnl[t] = pos * dx - trade_cost
        pos = new_pos

    return {"pnl": pnl, "z": z_arr}
