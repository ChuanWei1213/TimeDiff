"""AAPL 報酬分布比較圖。

六個方法 + 真實 = 7 條序列。依 dataviz 規範類別色上限為 3，
故用 small multiples：每格單一方法與真實對照，不以顏色區分方法。

三張圖各回答不同問題：
  1 density  : 分布長什麼樣（線性縱軸，經典鐘形）—— 看振幅
  2 standard : 標準化後的形狀 vs N(0,1) —— 看形狀（排除振幅差異）
  3 tail     : 對數縱軸 —— 看尾部
"""
import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
from scipy import stats as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluation import loaders, config as C, report as RP

OUT = "results/yearly_report/figures"
TXT = "results/yearly_report/tables"
os.makedirs(OUT, exist_ok=True)
METHODS = [("dow10_wf_v2","TimeDiff(Baseline)"),("dow10_wf_vpred","TimeDiff(v-prediction)"),
           ("dva","DVA"),("cndiff","CN-Diff"),("nsdiff","NS-Diff"),("armd","ARMD")]
S_, E_ = "2018-01-02", "2024-12-30"
AAPL = C.TICKERS.index("AAPL")
BLUE, GREY = RP.SERIES[0], "#9a9992"

rd, R = loaders.load_real(S_, E_)
real = R[:, AAPL]
disc = loaders.discover_models(verbose=False)
data = {}
for tag, disp in METHODS:
    m = [x for x in disc if x["tag"] == tag]
    if m:
        _, S = loaders.load_synthetic(m[0], S_, E_, real_dates=rd)
        data[disp] = S[0][:, AAPL]

def save(fig, name, txt, cap):
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    open(f"{TXT}/{name}.txt", "w").write(f"# {name}\n{cap}\n{'='*72}\n\n{txt}\n")

def panels(title, sub, xlim, ylab, plot_fn, name, txt, cap, logy=False):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True, sharey=True)
    for ax, (disp, x) in zip(axes.flat, data.items()):
        plot_fn(ax, x)
        ax.set_title(disp, fontsize=10.5, fontweight="bold", loc="left", color=RP.INK_PRIMARY, pad=8)
        ax.set_xlim(*xlim)
        ax.grid(axis="y", alpha=.55); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=RP.INK_SECONDARY, loc="upper right")
        if logy: ax.set_yscale("log")
    for ax in axes[1]: ax.set_xlabel(sub, fontsize=9)
    for ax in axes[:, 0]: ax.set_ylabel(ylab, fontsize=9)
    fig.suptitle(title, fontsize=12.5, fontweight="bold", x=.005, ha="left", y=1.005)
    fig.tight_layout()
    save(fig, name, txt, cap)

# ── 圖 1：經典鐘形（線性縱軸，填色）──
g1 = np.linspace(-0.07, 0.07, 600)
kr = st.gaussian_kde(real)(g1)
def f1(ax, x):
    ax.fill_between(g1, kr, color=GREY, alpha=.35, lw=0, zorder=2)
    ax.plot(g1, kr, color=GREY, lw=2, label="Real", zorder=3)
    k = st.gaussian_kde(x)(g1)
    ax.fill_between(g1, k, color=BLUE, alpha=.30, lw=0, zorder=4)
    ax.plot(g1, k, color=BLUE, lw=2, label="Synthetic", zorder=5)
t1 = ["AAPL daily log-return density (2018-01-02 ~ 2024-12-30, T=1760)", "",
      f"{'series':<26}{'std':>10}{'skew':>9}{'exKurt':>9}"]
t1.append(f"{'REAL':<26}{real.std():>10.5f}{st.skew(real):>9.3f}{st.kurtosis(real):>9.2f}")
for d_, x in data.items():
    t1.append(f"{d_:<26}{x.std():>10.5f}{st.skew(x):>9.3f}{st.kurtosis(x):>9.2f}")
panels("AAPL return distribution — each method vs Real", "daily log return", (-0.07, 0.07),
       "density", f1, "aapl_return_density", "\n".join(t1),
       "Grey = Real, Blue = method. Linear y-axis; narrower curve = amplitude too small.")

# ── 圖 2：標準化後的形狀 vs N(0,1)（排除振幅差異，只比形狀）──
g2 = np.linspace(-5, 5, 600)
zr = st.gaussian_kde((real-real.mean())/real.std())(g2)
norm = st.norm.pdf(g2)
def f2(ax, x):
    z = (x-x.mean())/x.std()
    ax.plot(g2, norm, color="#c9c8c1", lw=1.6, ls="--", label="N(0,1)", zorder=2)
    ax.plot(g2, zr, color=GREY, lw=2, label="Real", zorder=3)
    k = st.gaussian_kde(z)(g2)
    ax.fill_between(g2, k, color=BLUE, alpha=.28, lw=0, zorder=4)
    ax.plot(g2, k, color=BLUE, lw=2, label="Synthetic", zorder=5)
t2 = ["AAPL standardized returns z=(r-mu)/sigma — shape only, amplitude removed", "",
      f"{'series':<26}{'skew':>9}{'exKurt':>9}   (N(0,1): 0.000 / 0.00)"]
t2.append(f"{'REAL':<26}{st.skew(real):>9.3f}{st.kurtosis(real):>9.2f}")
for d_, x in data.items():
    t2.append(f"{d_:<26}{st.skew(x):>9.3f}{st.kurtosis(x):>9.2f}")
panels("AAPL standardized return shape — vs Real and vs N(0,1)", "z = (r - mean) / std", (-5, 5),
       "density", f2, "aapl_return_shape", "\n".join(t2),
       "Amplitude removed. Dashed = standard normal; grey = Real; blue = method. "
       "Real is peakier and has fatter tails than N(0,1) — that is the stylized fact to reproduce.")

# ── 圖 3：尾部（對數縱軸）──
g3 = np.linspace(-0.13, 0.13, 600)
kr3 = st.gaussian_kde(real)(g3)
def f3(ax, x):
    ax.plot(g3, kr3, color=GREY, lw=2.2, label="Real", zorder=2)
    ax.plot(g3, st.gaussian_kde(x)(g3), color=BLUE, lw=2, label="Synthetic", zorder=3)
    ax.set_ylim(1e-2, 2e2)
t3 = ["AAPL tails (log y-axis)", "",
      f"{'series':<26}{'min':>10}{'max':>10}{'q01':>10}{'q99':>10}"]
t3.append(f"{'REAL':<26}{real.min():>10.4f}{real.max():>10.4f}"
          f"{np.quantile(real,.01):>10.4f}{np.quantile(real,.99):>10.4f}")
for d_, x in data.items():
    t3.append(f"{d_:<26}{x.min():>10.4f}{x.max():>10.4f}"
              f"{np.quantile(x,.01):>10.4f}{np.quantile(x,.99):>10.4f}")
panels("AAPL return tails — log y-axis", "daily log return", (-0.13, 0.13),
       "density (log)", f3, "aapl_return_tails", "\n".join(t3),
       "Log y-axis. Where the blue curve stops is the widest event that method ever produced.",
       logy=True)
print("三張圖已產生")
