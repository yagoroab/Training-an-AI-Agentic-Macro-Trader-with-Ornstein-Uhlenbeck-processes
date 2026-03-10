import numpy as np
from src.strategies.ou_threshold import ou_zscore, banded_position_from_z


def backtest_ou(
    x: np.ndarray,
    traded_price: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
    sigma: np.ndarray,
    cost_bps: float = 1.0,
    z_entry: float = 1.25,
    z_exit: float = 0.35,
    z_cap: float = 3.0,
    exposure: float = 1.00,
    max_leverage: float = 1.25,
    vxx_short_bias: float = 0.30,
    vol_target: float = 0.012,
    vol_window: int = 20,
    max_vol_mult: float = 1.75,
    vol_mult_floor: float = 0.60,
    kappa_min: float = 0.01,
    hl_max: float = 80.0,
    pos_ema_alpha: float = 1.0,
    rebalance_thresh: float = 0.10,
    carry_bps_per_day: float = 0.2,
    cash_yield_annual: float = 0.02,
    min_hold_days: int = 7,
    stop_loss: float = 0.12,
) -> dict:

    x = np.asarray(x, dtype=float)
    traded_price = np.asarray(traded_price, dtype=float)
    mu = np.asarray(mu, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    n = len(x)

    trade_cost_rate = float(cost_bps) * 1e-4
    carry_cost_rate = float(carry_bps_per_day) * 1e-4
    daily_cash_yield = (1.0 + float(cash_yield_annual)) ** (1.0 / 252.0) - 1.0

    traded_ret = np.zeros(n)

    for t in range(1, n):
        if traded_price[t - 1] > 0:
            traded_ret[t] = (traded_price[t] / traded_price[t - 1]) - 1.0

    pos = 0.0
    holding_days = 0
    entry_price = np.nan

    pnl = np.zeros(n)
    z_arr = np.full(n, np.nan)
    pos_arr = np.zeros(n)
    vol_mult_arr = np.zeros(n)
    risky_weight_arr = np.zeros(n)
    cash_weight_arr = np.zeros(n)
    cash_yield_arr = np.zeros(n)

    for t in range(1, n):

        start = max(1, t - int(vol_window))

        if (t - start) >= 2:
            vol_est = np.std(traded_ret[start:t], ddof=1)
        else:
            vol_est = np.nan

        if not np.isfinite(vol_est) or vol_est <= 1e-12:
            vol_mult = 0.0
        else:
            vol_mult = vol_target / vol_est
            vol_mult = np.clip(vol_mult, vol_mult_floor, max_vol_mult)

        vol_mult_arr[t] = vol_mult
        ret = traded_ret[t]

        mu_tm1 = mu[t - 1]
        kappa_tm1 = kappa[t - 1]
        sigma_tm1 = sigma[t - 1]

        target_pos = pos

        if np.isfinite(mu_tm1) and np.isfinite(kappa_tm1) and np.isfinite(sigma_tm1):

            if kappa_tm1 < kappa_min:

                target_pos = 0.0

            else:

                half_life = np.log(2) / max(1e-12, kappa_tm1)

                if half_life > hl_max:

                    target_pos = 0.0

                else:

                    z = ou_zscore(x[t - 1], mu_tm1, kappa_tm1, sigma_tm1)
                    z_arr[t] = z

                    target_pos = banded_position_from_z(
                        z=z,
                        prev_pos=pos,
                        z_entry=z_entry,
                        z_exit=z_exit,
                        z_cap=z_cap,
                        max_leverage=max_leverage,
                        vxx_short_bias=vxx_short_bias,
                    )

        else:

            target_pos = 0.0

        # -------------------------
        # MIN HOLDING PERIOD LOGIC
        # -------------------------

        if holding_days < min_hold_days and pos != 0:
            target_pos = pos

        # -------------------------
        # POSITION UPDATE
        # -------------------------

        if abs(target_pos - pos) < rebalance_thresh:
            new_pos = pos
        else:
            new_pos = (1.0 - pos_ema_alpha) * pos + pos_ema_alpha * target_pos

        # -------------------------
        # DETECT NEW TRADE
        # -------------------------

        if pos == 0 and new_pos != 0:
            holding_days = 0
            entry_price = traded_price[t]

        if new_pos != 0:
            holding_days += 1
        else:
            holding_days = 0
            entry_price = np.nan

        # -------------------------
        # HARD STOP LOSS
        # -------------------------

        if new_pos != 0 and np.isfinite(entry_price):

            pnl_since_entry = (traded_price[t] / entry_price) - 1.0

            if new_pos < 0:
                pnl_since_entry *= -1

            if pnl_since_entry <= -stop_loss:

                new_pos = 0.0
                holding_days = 0
                entry_price = np.nan

        # -------------------------
        # PORTFOLIO WEIGHTS
        # -------------------------

        risky_weight = np.clip(exposure * vol_mult * abs(new_pos), 0.0, max_leverage)

        cash_weight = max(0.0, 1.0 - risky_weight)

        cash_yield = cash_weight * daily_cash_yield

        trade_cost = trade_cost_rate * abs(new_pos - pos)

        carry_cost = carry_cost_rate * abs(new_pos)

        risky_leg = risky_weight * np.sign(new_pos) * ret

        pnl[t] = risky_leg + cash_yield - trade_cost - carry_cost

        pos = new_pos

        pos_arr[t] = pos
        risky_weight_arr[t] = risky_weight
        cash_weight_arr[t] = cash_weight
        cash_yield_arr[t] = cash_yield

    return {
        "pnl": pnl,
        "z": z_arr,
        "pos": pos_arr,
        "vol_mult": vol_mult_arr,
        "risky_weight": risky_weight_arr,
        "cash_weight": cash_weight_arr,
        "cash_yield": cash_yield_arr,
        "traded_ret": traded_ret,
    }