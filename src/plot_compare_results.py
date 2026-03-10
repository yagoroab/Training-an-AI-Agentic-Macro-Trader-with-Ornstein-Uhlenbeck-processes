from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_series(obj, name: str) -> pd.Series:
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"{name} must be a Series or 1-column DataFrame, got shape {obj.shape}")
        s = obj.iloc[:, 0].copy()
    else:
        raise TypeError(f"{name} must be a pandas Series or DataFrame, got {type(obj)}")

    s = s.astype(float)
    s.name = name
    return s.sort_index()


def _align_series(ou_wealth: pd.Series, benchmarks_wealth: dict[str, pd.Series]) -> pd.DataFrame:
    ou_wealth = _to_series(ou_wealth, "OU Strategy")
    df = pd.DataFrame({"OU Strategy": ou_wealth})

    for name, s in benchmarks_wealth.items():
        df[name] = _to_series(s, name)

    df = df.dropna(how="any").sort_index()
    return df


def _to_drawdown(wealth: pd.Series) -> pd.Series:
    w = _to_series(wealth, wealth.name if wealth.name is not None else "wealth")
    peak = w.cummax()
    dd = (w / peak) - 1.0
    dd.name = f"{w.name}_drawdown"
    return dd


def _daily_returns_from_wealth(wealth: pd.Series) -> pd.Series:
    w = _to_series(wealth, wealth.name if wealth.name is not None else "wealth")
    r = w.pct_change().fillna(0.0)
    r.name = f"{w.name}_ret"
    return r


@dataclass
class PerfStats:
    cagr: float
    vol_ann: float
    sharpe: float
    max_dd: float
    final_wealth: float


def _perf_stats(wealth: pd.Series, periods_per_year: int = 252) -> PerfStats:
    wealth = _to_series(wealth, wealth.name if wealth.name is not None else "wealth").dropna()

    if len(wealth) < 3:
        return PerfStats(cagr=0.0, vol_ann=0.0, sharpe=0.0, max_dd=0.0, final_wealth=0.0)

    t0 = wealth.index[0]
    t1 = wealth.index[-1]
    years = (t1 - t0).days / 365.25
    final = float(wealth.iloc[-1])

    cagr = final ** (1.0 / years) - 1.0 if years > 0 and final > 0 else 0.0

    r = _daily_returns_from_wealth(wealth)
    r_std = float(r.std(ddof=1))

    vol = r_std * np.sqrt(periods_per_year) if r_std > 0 else 0.0
    sharpe = (float(r.mean()) / r_std) * np.sqrt(periods_per_year) if r_std > 0 else 0.0
    max_dd = float(_to_drawdown(wealth).min())

    return PerfStats(
        cagr=float(cagr),
        vol_ann=float(vol),
        sharpe=float(sharpe),
        max_dd=float(max_dd),
        final_wealth=final,
    )


def plot_wealth_comparison(
    ou_wealth: pd.Series,
    benchmarks_wealth: dict[str, pd.Series],
    outpath: str,
    title: str,
    log_scale: bool = False,
) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df = _align_series(ou_wealth, benchmarks_wealth)

    plt.figure(figsize=(12, 4.5))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, linewidth=1.6)

    if log_scale:
        plt.yscale("log")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Wealth (normalized to 1.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_drawdown_comparison(
    ou_wealth: pd.Series,
    benchmarks_wealth: dict[str, pd.Series],
    outpath: str,
    title: str = "Drawdown Comparison",
) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    dfw = _align_series(ou_wealth, benchmarks_wealth)

    plt.figure(figsize=(12, 4.5))
    for col in dfw.columns:
        dd = _to_drawdown(dfw[col])
        plt.plot(dd.index, dd.values, label=col, linewidth=1.4)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_excess_vs_benchmark(
    ou_wealth: pd.Series,
    benchmark_wealth: pd.Series,
    outpath_ratio: str,
    outpath_log_ratio: str,
    benchmark_name: str = "Benchmark",
) -> None:
    os.makedirs(os.path.dirname(outpath_ratio), exist_ok=True)

    ou_s = _to_series(ou_wealth, "OU Strategy")
    bm_s = _to_series(benchmark_wealth, benchmark_name)

    df = pd.concat([ou_s.rename("OU Strategy"), bm_s.rename(benchmark_name)], axis=1)
    df = df.dropna().sort_index()

    ratio = df["OU Strategy"] / df[benchmark_name]
    ratio.name = "OU_over_benchmark"

    plt.figure(figsize=(12, 4.5))
    plt.plot(ratio.index, ratio.values, label=f"OU / {benchmark_name}", linewidth=1.6)
    plt.axhline(1.0, linestyle="--")
    plt.title(f"Relative Performance: OU Strategy vs {benchmark_name}")
    plt.xlabel("Date")
    plt.ylabel("Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_ratio, dpi=200)
    plt.close()

    log_ratio = np.log(ratio.replace(0.0, np.nan)).dropna()

    plt.figure(figsize=(12, 4.5))
    plt.plot(log_ratio.index, log_ratio.values, label=f"log(OU / {benchmark_name})", linewidth=1.6)
    plt.axhline(0.0, linestyle="--")
    plt.title(f"Log Relative Performance: OU Strategy vs {benchmark_name}")
    plt.xlabel("Date")
    plt.ylabel("log-ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_log_ratio, dpi=200)
    plt.close()


def print_annualized_table(
    ou_wealth: pd.Series,
    benchmarks_wealth: dict[str, pd.Series],
) -> None:
    df = _align_series(ou_wealth, benchmarks_wealth)

    rows = []
    for col in df.columns:
        st = _perf_stats(df[col])
        rows.append(
            {
                "Strategy": col,
                "CAGR": st.cagr,
                "Vol (ann.)": st.vol_ann,
                "Sharpe": st.sharpe,
                "Max Drawdown": st.max_dd,
                "Final Wealth": st.final_wealth,
            }
        )

    out = pd.DataFrame(rows).set_index("Strategy")

    out_fmt = out.copy()
    out_fmt["CAGR"] = out_fmt["CAGR"].map(lambda x: f"{x:.2%}")
    out_fmt["Vol (ann.)"] = out_fmt["Vol (ann.)"].map(lambda x: f"{x:.2%}")
    out_fmt["Sharpe"] = out_fmt["Sharpe"].map(lambda x: f"{x:.2f}")
    out_fmt["Max Drawdown"] = out_fmt["Max Drawdown"].map(lambda x: f"{x:.2%}")
    out_fmt["Final Wealth"] = out_fmt["Final Wealth"].map(lambda x: f"{x:.2f}")

    print("\n=== Annualized Performance (aligned sample) ===")
    print(out_fmt.to_string())