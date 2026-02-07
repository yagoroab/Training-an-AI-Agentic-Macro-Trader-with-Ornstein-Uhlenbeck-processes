import numpy as np

def fit_ou_ar1(x: np.ndarray, dt: float) -> dict:
    """
    Fit OU params using AR(1) OLS on exact discretization.
    Returns dict: mu, kappa, sigma.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return {"mu": np.nan, "kappa": np.nan, "sigma": np.nan}

    x0 = x[:-1]
    x1 = x[1:]

    # OLS: x1 = a + b*x0 + eps
    A = np.vstack([np.ones_like(x0), x0]).T
    a, b = np.linalg.lstsq(A, x1, rcond=None)[0]

    b = float(np.clip(b, 1e-3, 1 - 1e-4))
    kappa = -np.log(b) / dt
    mu = a / (1 - b)

    resid = x1 - (a + b * x0)
    var_eps = resid.var(ddof=2)

    sigma = np.sqrt(var_eps * 2 * kappa / (1 - b**2))
    return {"mu": float(mu), "kappa": float(kappa), "sigma": float(sigma)}

def rolling_ou_params(x: np.ndarray, window: int, dt: float):
    """
    Rolling OU estimation. Returns arrays (mu, kappa, sigma) aligned to x.
    NaN until t >= window.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    mu = np.full(n, np.nan)
    kappa = np.full(n, np.nan)
    sigma = np.full(n, np.nan)

    for t in range(window, n):
        est = fit_ou_ar1(x[t-window:t], dt=dt)
        mu[t] = est["mu"]
        kappa[t] = est["kappa"]
        sigma[t] = est["sigma"]

    return mu, kappa, sigma
