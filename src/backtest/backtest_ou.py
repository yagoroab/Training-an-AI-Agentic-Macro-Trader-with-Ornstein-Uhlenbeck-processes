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
    vxx_short_bias: float = 0.00,
    max_long_leverage: float = 0.50,
    vol_target: float = 0.014,
    vol_window: int = 20,
    max_vol_mult: float = 2.00,
    vol_mult_floor: float = 0.60,
    kappa_min: float = 0.01,
    hl_max: float = 80.0,
    pos_ema_alpha: float = 0.50,
    rebalance_thresh: float = 0.20,
    carry_bps_per_day: float = 0.2,
    cash_yield_annual: float = 0.03,
    min_hold_days: int = 10,
    stop_loss: float = 0.08,
    trend_window: int = 20,
    trend_buffer: float = 9.99,
    long_entry_scale: float = 1.0,
    confidence_kappa_ref: float = 0.08,
    confidence_floor: float = 1.0,
    stop_cooldown_days: int = 0,
    stress_vix_level: float = 1e9,
    crash_vix_level: float = 1e9,
    stress_jump: float = 1e9,
    crash_jump: float = 1e9,
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
    risky_weight = 0.0
    cash_weight = 1.0
    holding_days = 0
    entry_price = np.nan
    cooldown_days_left = 0

    pnl = np.zeros(n)
    z_arr = np.full(n, np.nan)
    pos_arr = np.zeros(n)
    vol_mult_arr = np.zeros(n)
    risky_weight_arr = np.zeros(n)
    cash_weight_arr = np.zeros(n)
    cash_yield_arr = np.zeros(n)
    cash_weight_arr[0] = cash_weight

    for t in range(1, n):
        ret = traded_ret[t]
        carry_cost = carry_cost_rate * abs(pos)
        cash_yield = cash_weight * daily_cash_yield
        risky_leg = risky_weight * np.sign(pos) * ret

        pnl[t] = risky_leg + cash_yield - carry_cost

        if pos != 0.0:
            holding_days += 1

        start = max(1, t - int(vol_window) + 1)

        if (t - start + 1) >= 2:
            vol_est = np.std(traded_ret[start : t + 1], ddof=1)
        else:
            vol_est = np.nan

        if not np.isfinite(vol_est) or vol_est <= 1e-12:
            vol_mult = 0.0
        else:
            vol_mult = vol_target / vol_est
            vol_mult = np.clip(vol_mult, vol_mult_floor, max_vol_mult)

        vol_mult_arr[t] = vol_mult

        mu_t = mu[t]
        kappa_t = kappa[t]
        sigma_t = sigma[t]
        vix_level_t = float(np.exp(x[t]))
        vix_level_tm1 = float(np.exp(x[t - 1]))
        vix_jump = (vix_level_t / max(vix_level_tm1, 1e-12)) - 1.0

        target_pos = pos
        z = np.nan

        if np.isfinite(mu_t) and np.isfinite(kappa_t) and np.isfinite(sigma_t):

            if kappa_t < kappa_min:

                target_pos = 0.0

            else:

                half_life = np.log(2) / max(1e-12, kappa_t)

                if half_life > hl_max:

                    target_pos = 0.0

                else:

                    z = ou_zscore(x[t], mu_t, kappa_t, sigma_t)
                    z_arr[t] = z

                    target_pos = banded_position_from_z(
                        z=z,
                        prev_pos=pos,
                        z_entry=z_entry,
                        z_exit=z_exit,
                        z_cap=z_cap,
                        max_leverage=max_leverage,
                        vxx_short_bias=vxx_short_bias,
                        max_long_leverage=max_long_leverage,
                    )

        else:

            target_pos = 0.0

        trend_start = max(0, t - int(trend_window) + 1)
        vix_trend_ma = float(np.mean(np.exp(x[trend_start : t + 1])))
        vxx_trend_ma = float(np.mean(traded_price[trend_start : t + 1]))

        vix_uptrend = vix_level_t > vix_trend_ma * (1.0 + trend_buffer)
        vxx_uptrend = traded_price[t] > vxx_trend_ma * (1.0 + trend_buffer)
        stress_state = (vix_level_t >= stress_vix_level) or (vix_jump >= stress_jump)
        crash_state = (vix_level_t >= crash_vix_level) or (vix_jump >= crash_jump)

        if np.isfinite(z):
            signal_strength = np.clip((abs(z) - z_exit) / max(1e-12, (z_cap - z_exit)), 0.0, 1.0)
        else:
            signal_strength = 0.0

        kappa_strength = np.clip(
            (kappa_t - kappa_min) / max(1e-12, (confidence_kappa_ref - kappa_min)),
            0.0,
            1.0,
        ) if np.isfinite(kappa_t) else 0.0
        confidence_mult = confidence_floor + (1.0 - confidence_floor) * signal_strength * kappa_strength
        target_pos *= confidence_mult

        if target_pos > 0.0 and (not np.isfinite(z) or z > -(long_entry_scale * z_entry)):
            target_pos = 0.0

        if target_pos < 0.0 and (vix_uptrend or vxx_uptrend or stress_state or cooldown_days_left > 0):
            target_pos = 0.0

        if crash_state:
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
        # HARD STOP LOSS
        # -------------------------

        if pos != 0 and np.isfinite(entry_price):

            pnl_since_entry = (traded_price[t] / entry_price) - 1.0

            if pos < 0:
                pnl_since_entry *= -1

            if pnl_since_entry <= -stop_loss:

                new_pos = 0.0
                cooldown_days_left = max(cooldown_days_left, int(stop_cooldown_days))

        # -------------------------
        # PORTFOLIO WEIGHTS FOR t+1
        # -------------------------

        next_risky_weight = np.clip(exposure * vol_mult * abs(new_pos), 0.0, max_leverage)

        next_cash_weight = max(0.0, 1.0 - next_risky_weight)

        trade_cost = trade_cost_rate * abs(new_pos - pos)
        pnl[t] -= trade_cost

        if new_pos == 0.0:
            holding_days = 0
            entry_price = np.nan
        elif pos == 0.0 or np.sign(new_pos) != np.sign(pos):
            holding_days = 0
            entry_price = traded_price[t]

        pos = new_pos
        risky_weight = next_risky_weight
        cash_weight = next_cash_weight

        pos_arr[t] = pos
        risky_weight_arr[t] = risky_weight
        cash_weight_arr[t] = cash_weight
        cash_yield_arr[t] = cash_yield

        if cooldown_days_left > 0:
            cooldown_days_left -= 1

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
