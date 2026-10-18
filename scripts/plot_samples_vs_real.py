"""Visual check: random TimeDiff RL paths vs real history.

    python scripts/plot_samples_vs_real.py \
        --paths outputs/dow10/conditional/fixed_scenarios/legacy/rl_paths/paths.npz \
        --data datasets/dow10/prices_2004.csv \
        --out outputs/dow10/conditional/fixed_scenarios/legacy/figures

Figures (equal-weight 10-stock portfolio unless noted, start of window = 1):
  1_examples.png       per scenario, 4 random episodes: 50 real days + 21 generated days,
                       portfolio (thick) and the 10 stocks (thin)
  2_overlay.png        per scenario: 40 random generated 21-day paths vs the real 21-day
                       windows (2005-2017) whose return/vol are closest to the generated mean,
                       plus random ordinary real windows
  3_distributions.png  daily stock returns, 21-day realized vol, 21-day return and mean
                       pairwise correlation: generated scenarios vs real windows
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMES = {0: "S1 (-10%/40%)", 1: "S2 (-20%/60%)", 2: "S3 (-30%/80%)"}
COLORS = {0: "tab:orange", 1: "tab:red", 2: "darkred"}


def ew_path(close_rel):
    """close_rel [..., T, A] prices relative to start -> daily-rebalanced EW value [..., T]."""
    gross = close_rel[..., 1:, :] / close_rel[..., :-1, :]
    v = np.cumprod(gross.mean(axis=-1), axis=-1)
    return np.concatenate([np.ones(v.shape[:-1] + (1,)), v], axis=-1)


def window_stats(rc):
    """rc [n, 21, A] daily log returns -> R, vol (EW portfolio), mean pairwise corr."""
    rp = np.log(np.exp(rc).mean(axis=-1))
    R = np.expm1(rp.sum(axis=1))
    V = rp.std(axis=1, ddof=1) * np.sqrt(252)
    A = rc.shape[-1]
    mask = ~np.eye(A, dtype=bool)
    C = np.array([np.corrcoef(x.T)[mask].mean() if np.all(x.std(0) > 0) else np.nan for x in rc])
    return R, V, C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--real-start", default="2005-01-01")
    ap.add_argument("--real-end", default="2017-12-31")
    ap.add_argument("--obs-days", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(cli.seed)
    cli.out.mkdir(parents=True, exist_ok=True)
    z = np.load(cli.paths, allow_pickle=False)
    tickers = [str(t) for t in z["tickers"]]
    prices = pd.read_csv(cli.data, parse_dates=["date"])
    close = prices.pivot(index="date", columns="tic", values="close")[tickers]
    dates = close.index
    C = close.to_numpy(float)
    syn_close, scen = z["close"].astype(float), z["scenario"]
    anchors = np.array([dates.get_loc(pd.Timestamp(str(d))) for d in z["anchor_dates"]])
    scenarios = sorted(set(scen.tolist()))

    # ---------- real 21-day windows (for comparison) ----------
    lo, hi = dates.searchsorted(pd.Timestamp(cli.real_start)), dates.searchsorted(pd.Timestamp(cli.real_end), "right") - 1
    starts = np.arange(max(lo, 1), hi - 21)
    logC = np.log(C)
    real_rc = np.stack([np.diff(logC[s - 1:s + 21], axis=0) for s in starts])        # [n, 21, A]
    real_R, real_V, real_C = window_stats(real_rc)
    syn_rc = np.diff(np.log(np.concatenate([C[anchors][:, None, :], syn_close], axis=1)), axis=1)
    syn_R, syn_V, syn_C = window_stats(syn_rc)

    # ---------- 1. example episodes ----------
    W, ncol = cli.obs_days, 4
    fig, axes = plt.subplots(len(scenarios), ncol, figsize=(4 * ncol, 3.2 * len(scenarios)), squeeze=False)
    for r, sc in enumerate(scenarios):
        for c, j in enumerate(rng.choice(np.flatnonzero(scen == sc), ncol, replace=False)):
            ax, a = axes[r, c], anchors[j]
            hist = C[a - W + 1:a + 1] / C[a]
            fut = syn_close[j] / C[a]
            full = np.vstack([hist, fut])
            x = np.arange(-W + 1, 22)
            ax.plot(x, full, color="gray", lw=.5, alpha=.6)
            ew = ew_path(full / full[0])
            ew = ew / ew[W - 1]
            ax.plot(x[:W], ew[:W], color="black", lw=2)
            ax.plot(x[W - 1:], ew[W - 1:], color=COLORS.get(sc, "tab:red"), lw=2.2)
            ax.axvline(0, color="k", ls="--", lw=.8)
            ax.set_title(f"{NAMES.get(sc, sc)}  anchor {str(z['anchor_dates'][j])}\n"
                         f"gen R {syn_R[j]:+.1%}  vol {syn_V[j]:.0%}  corr {syn_C[j]:.2f}", fontsize=9)
            ax.grid(alpha=.3)
    fig.suptitle("Black: real 50-day history (portfolio). Colored: generated 21 days. Gray: 10 stocks. "
                 "Prices / anchor close.", fontsize=10)
    fig.tight_layout()
    fig.savefig(cli.out / "1_examples.png", dpi=130)
    plt.close(fig)

    # ---------- 2. overlay generated vs nearest real stress windows ----------
    fig, axes = plt.subplots(1, len(scenarios) + 1, figsize=(4.4 * (len(scenarios) + 1), 4), sharey=True)
    real_paths = np.stack([ew_path(C[s - 1:s + 21] / C[s - 1]) for s in starts])     # [n, 22]
    ax = axes[0]
    for s in rng.choice(len(starts), 40, replace=False):
        ax.plot(real_paths[s], color="tab:blue", alpha=.35, lw=1)
    ax.set_title(f"Real, random 21-day windows\n({cli.real_start[:4]}-{cli.real_end[:4]})", fontsize=10)
    for i, sc in enumerate(scenarios, start=1):
        ax = axes[i]
        m = scen == sc
        for j in rng.choice(np.flatnonzero(m), 40, replace=False):
            p = ew_path(np.vstack([C[anchors[j]], syn_close[j]]) / C[anchors[j]])
            ax.plot(p, color=COLORS.get(sc, "tab:red"), alpha=.3, lw=1)
        tgt = np.array([syn_R[m].mean(), syn_V[m].mean()])
        d = np.hypot((real_R - tgt[0]) / real_R.std(), (real_V - tgt[1]) / real_V.std())
        near = np.argsort(d)[:10]
        for s in near:
            ax.plot(real_paths[s], color="black", lw=1.6, alpha=.8)
        yrs = sorted({str(dates[starts[s]].year) for s in near})
        ax.set_title(f"{NAMES.get(sc, sc)}: generated (color)\nvs 10 closest real windows (black, {', '.join(yrs)})",
                     fontsize=10)
    for ax in axes:
        ax.axhline(1, color="k", lw=.6)
        ax.set_xlabel("trading day")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("EW portfolio value (start = 1)")
    fig.tight_layout()
    fig.savefig(cli.out / "2_overlay.png", dpi=130)
    plt.close(fig)

    # ---------- 3. distributions ----------
    top = np.argsort(real_V)[-50:]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    bins = np.linspace(-0.25, 0.25, 101)
    axes[0].hist(real_rc.ravel(), bins=bins, density=True, histtype="step", color="tab:blue", label="real all")
    axes[0].hist(real_rc[top].ravel(), bins=bins, density=True, histtype="step", color="black", label="real top-50 vol")
    for sc in scenarios:
        axes[0].hist(syn_rc[scen == sc].ravel(), bins=bins, density=True, histtype="step",
                     color=COLORS.get(sc), label=NAMES.get(sc, sc))
    axes[0].set_yscale("log")
    axes[0].set_title("Daily stock log returns (log density)")
    for k, (vals_real, vals_syn, title, rng_) in enumerate([
            (real_V, syn_V, "21-day EW portfolio vol", (0, 1.2)),
            (real_R, syn_R, "21-day EW portfolio return", (-0.45, 0.25)),
            (real_C, syn_C, "mean pairwise correlation (21 days)", (-0.2, 1))], start=1):
        b = np.linspace(*rng_, 60)
        axes[k].hist(vals_real, bins=b, density=True, alpha=.35, color="tab:blue", label="real all")
        axes[k].hist(vals_real[top], bins=b, density=True, histtype="step", lw=2, color="black", label="real top-50 vol")
        for sc in scenarios:
            axes[k].hist(vals_syn[scen == sc], bins=b, density=True, histtype="step", lw=1.5,
                         color=COLORS.get(sc), label=NAMES.get(sc, sc))
        axes[k].set_title(title)
    for ax in axes:
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cli.out / "3_distributions.png", dpi=130)
    plt.close(fig)

    print(f"real windows: n={len(starts)}, worst R {real_R.min():+.1%} ({dates[starts[real_R.argmin()]].date()}), "
          f"max vol {real_V.max():.0%} ({dates[starts[real_V.argmax()]].date()})")
    for sc in scenarios:
        m = scen == sc
        print(f"{NAMES.get(sc, sc)}: R {syn_R[m].mean():+.1%}  vol {syn_V[m].mean():.0%}  corr {np.nanmean(syn_C[m]):.2f}  "
              f"| daily stock |r|>10%: {np.mean(np.abs(syn_rc[m]) > 0.10):.2%} (real top-50 vol: "
              f"{np.mean(np.abs(real_rc[top]) > 0.10):.2%})")
    print(f"figures in {cli.out}")


if __name__ == "__main__":
    main()