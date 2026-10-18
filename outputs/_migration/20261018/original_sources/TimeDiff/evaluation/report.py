"""圖表輸出：每張圖同時存 PNG 與一份含完整數值的 txt。

配色依 dataviz 規範：
  - 分數 / 尾部相依（純量值，magnitude）→ sequential 單一藍色 ramp（淺→深）
  - 相關係數（有極性，[-1,1]）→ diverging 藍↔紅 + 中性灰中點
  - 類別序列上限 3 個（藍/橘/青，all-pairs CVD 驗證通過）；超過 3 類改用 small multiples
文字一律使用 ink 顏色而非序列色；格線與軸線退居背景。
"""
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

# ── 設計 token ──────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e1"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]        # 藍 / 橘 / 青

# sequential 藍 ramp（palette.md 的 100→700 步階）
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
# diverging 藍↔紅，中性灰中點
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8f1f1e"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "text.color": INK_PRIMARY, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.grid": False, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _ensure_dirs():
    for d in (C.FIG_DIR, C.FIGDATA_DIR, C.TABLE_DIR):
        os.makedirs(d, exist_ok=True)


def save_figure(fig, name, data_text, caption=""):
    """存 PNG + 同名 txt（txt 內含產生該圖所用的完整數值）。"""
    _ensure_dirs()
    png = os.path.join(C.FIG_DIR, f"{name}.png")
    txt = os.path.join(C.FIGDATA_DIR, f"{name}.txt")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(txt, "w") as f:
        f.write(f"# {name}\n")
        if caption:
            f.write(textwrap.fill(caption, 100) + "\n")
        f.write("#" + "=" * 79 + "\n\n")
        f.write(data_text.rstrip() + "\n")
    return png, txt


def _annotate_cells(ax, M, fmt="{:.3f}", thresh=0.55, vmin=0.0, vmax=1.0):
    """在格子上寫值；文字用 ink 色，深底改白字以維持對比。"""
    rng = (vmax - vmin) or 1.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "N/A", ha="center", va="center", color=INK_MUTED, fontsize=8)
                continue
            rel = (v - vmin) / rng
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="#ffffff" if rel > thresh else INK_PRIMARY)


# ===========================================================
# FIGURE 1 / 2 — Score heatmaps
# ===========================================================
def fig_score_heatmap(df, name, title, subtitle, col_labels=None, flags=None):
    """df: index = model, columns = metric, values = score in [0,1]。

    分數是 magnitude（0=差、1=完美），故用單一藍 ramp，不用彩虹。
    """
    M = df.values.astype(float)
    cols = col_labels or list(df.columns)
    n_r, n_c = len(df), len(cols)
    cell = 1.15                                   # 每格邊長（吋），維持方形格
    fig, ax = plt.subplots(figsize=(3.6 + cell * n_c, 1.9 + cell * n_r))
    im = ax.imshow(M, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n_c))
    ax.set_xticklabels([textwrap.fill(c, 12) for c in cols], fontsize=8)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(df.index, fontsize=9)
    ax.tick_params(axis="x", pad=6, length=0)
    ax.tick_params(axis="y", length=0)
    _annotate_cells(ax, M)
    ax.set_xticks(np.arange(-.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_r, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)   # 2px surface gap
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, shrink=0.9)
    cb.set_label("Fidelity score  (1 = matches real)", fontsize=8, color=INK_SECONDARY)
    cb.outline.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK_PRIMARY, loc="left", pad=26)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8,
            color=INK_SECONDARY, va="bottom")
    if flags:
        # 放在整張圖的最下緣，避開 x 軸標籤（bbox_inches='tight' 會納入）
        fig.text(0.01, -0.02 - 0.035 * (len(flags) - 1), "\n".join(flags),
                 fontsize=7.5, color=INK_MUTED, va="top", ha="left")
    txt = df.round(4).to_string()
    if flags:
        txt += "\n\n" + "\n".join(flags)
    return fig, txt


# ===========================================================
# FIGURE 3 — PCA explained variance
# ===========================================================
def fig_pca_spectrum(evr_real, evr_syn_by_model):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    k = np.arange(1, len(evr_real) + 1)
    ax.plot(k, evr_real, "-o", color=SERIES[0], lw=2, ms=8, label="Real", zorder=3)
    for n, (mname, evr) in enumerate(evr_syn_by_model.items(), start=1):
        ax.plot(k, evr, "-o", color=SERIES[n % len(SERIES)], lw=2, ms=8,
                label=mname, zorder=3)
    ax.set_xticks(k)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    ax.set_title("Common factor structure: real vs synthetic",
                 fontsize=11, fontweight="bold", loc="left", pad=12)
    ax.text(0, 1.02, "Correlation-matrix eigenvalues. PC1 = market factor strength.",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="bottom")
    # 只對 PC1 直接標值（selective direct labels）
    ax.annotate(f"{evr_real[0]:.3f}", (1, evr_real[0]), textcoords="offset points",
                xytext=(6, 8), fontsize=8, color=INK_SECONDARY)
    for mname, evr in evr_syn_by_model.items():
        ax.annotate(f"{evr[0]:.3f}", (1, evr[0]), textcoords="offset points",
                    xytext=(6, -14), fontsize=8, color=INK_SECONDARY)

    lines = ["PC\tReal\t" + "\t".join(evr_syn_by_model.keys())]
    for i in range(len(evr_real)):
        lines.append(f"PC{i+1}\t{evr_real[i]:.6f}\t" +
                     "\t".join(f"{v[i]:.6f}" for v in evr_syn_by_model.values()))
    return fig, "\n".join(lines)


# ===========================================================
# FIGURE 4 — Correlation matrices
# ===========================================================
def fig_corr_matrices(C_real, C_syn, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    D = C_syn - C_real
    for ax, M, ttl, cmap, vlim in [
            (axes[0], C_real, "Real", CMAP_DIV, 1.0),
            (axes[1], C_syn, f"Synthetic — {model_name}", CMAP_DIV, 1.0),
            (axes[2], D, "Difference (syn − real)", CMAP_DIV, float(np.nanmax(np.abs(D))))]:
        im = ax.imshow(M, cmap=cmap, vmin=-vlim, vmax=vlim)
        ax.set_xticks(range(C.N_TICKER)); ax.set_xticklabels(C.TICKERS, rotation=90, fontsize=7.5)
        ax.set_yticks(range(C.N_TICKER)); ax.set_yticklabels(C.TICKERS, fontsize=7.5)
        ax.set_title(ttl, fontsize=10, fontweight="bold", loc="left", color=INK_PRIMARY)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(wspace=0.55)
    fig.suptitle("Cross-asset correlation structure", fontsize=12, fontweight="bold",
                 x=0.005, ha="left", y=1.02)
    lines = ["# Real correlation matrix", pd.DataFrame(C_real, C.TICKERS, C.TICKERS).round(4).to_string(),
             "", f"# Synthetic ({model_name})",
             pd.DataFrame(C_syn, C.TICKERS, C.TICKERS).round(4).to_string(),
             "", "# Difference (syn - real)",
             pd.DataFrame(D, C.TICKERS, C.TICKERS).round(4).to_string()]
    return fig, "\n".join(lines)


# ===========================================================
# FIGURE 5 — Lower tail dependence matrices
# ===========================================================
def fig_tail_dep_matrices(T_real, T_syn, model_name, q):
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    D = T_syn - T_real
    for ax, M, ttl, cmap, kw in [
            (axes[0], T_real, "Real", CMAP_SEQ, dict(vmin=0, vmax=1)),
            (axes[1], T_syn, f"Synthetic — {model_name}", CMAP_SEQ, dict(vmin=0, vmax=1)),
            (axes[2], D, "Difference (syn − real)", CMAP_DIV,
             dict(vmin=-np.nanmax(np.abs(D)), vmax=np.nanmax(np.abs(D))))]:
        im = ax.imshow(M, cmap=cmap, **kw)
        ax.set_xticks(range(C.N_TICKER)); ax.set_xticklabels(C.TICKERS, rotation=90, fontsize=7.5)
        ax.set_yticks(range(C.N_TICKER)); ax.set_yticklabels(C.TICKERS, fontsize=7.5)
        ax.set_title(ttl, fontsize=10, fontweight="bold", loc="left", color=INK_PRIMARY)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(wspace=0.55)
    fig.suptitle(f"Lower-tail dependence  P(U_j < {q} | U_i < {q})  — copula / rank based",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.02)
    lines = ["# Real lower-tail dependence",
             pd.DataFrame(T_real, C.TICKERS, C.TICKERS).round(4).to_string(),
             "", f"# Synthetic ({model_name})",
             pd.DataFrame(T_syn, C.TICKERS, C.TICKERS).round(4).to_string(),
             "", "# Difference (syn - real)",
             pd.DataFrame(D, C.TICKERS, C.TICKERS).round(4).to_string()]
    return fig, "\n".join(lines)


# ===========================================================
# FIGURE 6 — Overall model comparison
# ===========================================================
def fig_overall(df):
    """df: index = model, columns = [Stylized Facts, Cross-Asset, Overall]."""
    models = list(df.index)
    cols = list(df.columns)
    n, m = len(models), len(cols)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.9 * n + 3), 4.2))
    w = 0.8 / m
    for j, c in enumerate(cols):
        pos = np.arange(n) + (j - (m - 1) / 2) * w
        bars = ax.bar(pos, df[c].values, width=w * 0.92, color=SERIES[j],
                      label=c, zorder=3)
        for b, v in zip(bars, df[c].values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(n))
    ax.set_xticklabels([textwrap.fill(x, 22) for x in models], fontsize=9)
    ax.set_ylabel("Fidelity score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=m)
    ax.set_title("Overall financial fidelity", fontsize=11, fontweight="bold",
                 loc="left", pad=12)
    ax.text(0, 1.02, f"Primary evaluation window {C.PRIMARY_EVAL_START} ~ {C.PRIMARY_EVAL_END}. "
                     "1 = indistinguishable from real.",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="bottom")
    return fig, df.round(4).to_string()


# ===========================================================
# FIGURE 7 — Asset-level score distribution（small multiples）
# ===========================================================
def fig_asset_level(asset_df, metric_labels):
    """5 個 metric 各一格，每格顯示 10 檔的分數。

    刻意不用 5 種顏色編碼 metric（超過 3 個類別色會失去 CVD 安全性），
    改用 small multiples，每格單一序列、不需圖例。
    """
    metrics = list(metric_labels.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 4.4), sharey=True)
    lines = []
    for ax, m in zip(np.atleast_1d(axes), metrics):
        sub = asset_df[asset_df.metric == m]
        g = sub.groupby("ticker")["score"]
        mean = g.mean().reindex(C.TICKERS)
        lo, hi = g.min().reindex(C.TICKERS), g.max().reindex(C.TICKERS)
        y = np.arange(C.N_TICKER)
        ax.hlines(y, lo, hi, color=SEQ_BLUE[2], lw=3, zorder=2)      # 跨路徑全距
        ax.plot(mean.values, y, "o", color=SERIES[0], ms=8, zorder=3)
        ax.axvline(float(mean.mean()), color=INK_MUTED, lw=1, ls="--", zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(C.TICKERS, fontsize=8)
        ax.invert_yaxis()                       # AAPL 在最上，與表格順序一致
        ax.set_xlim(0, 1); ax.set_xlabel("score")
        ax.set_title(textwrap.fill(metric_labels[m], 18), fontsize=9,
                     fontweight="bold", loc="left", color=INK_PRIMARY, pad=10)
        ax.grid(axis="x", alpha=0.9); ax.set_axisbelow(True)
        lines.append(f"# {metric_labels[m]}  (10-ticker mean = {mean.mean():.4f})")
        lines.append(pd.DataFrame({"mean": mean, "min": lo, "max": hi}).round(4).to_string())
        lines.append("")
    fig.suptitle("Asset-level score dispersion — is the average carried by a few tickers?",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.10)
    fig.text(0.005, 1.055, "Dot = mean across generated paths; bar = min–max across paths. "
                           "Dashed line = 10-ticker average.",
             fontsize=8, color=INK_SECONDARY, ha="left", va="top")
    fig.subplots_adjust(top=0.86)
    return fig, "\n".join(lines)
