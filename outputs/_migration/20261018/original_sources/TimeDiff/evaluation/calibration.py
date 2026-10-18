"""Normalization / Score design — 以「真實資料自身的重抽噪音」作為距離的尺度。

核心思想
--------
每個 metric 的距離 D 要轉成 [0,1] 的 score，必須先知道「D 多大才算大」。
本框架用真實資料的 stationary block bootstrap 求出

    tau_m = E_b [ D_m( theta_hat(X_real^{b1}), theta_hat(X_real^{b2}) ) ]

也就是「即使模型完美無誤，兩次獨立觀測之間也必然存在的距離」。然後

    Score = 2 ** (-D / tau)

性質：D=0 -> 1；D=tau -> 0.5；D->inf -> 0。tau **只用真實資料**計算，
不接觸任何 synthetic 輸出，因此不存在 circularity。

為什麼不用其他做法
------------------
  score = 1 - D        : 需要 D in [0,1]，只有 TV distance 滿足
  score = exp(-D)      : 等於把 tau 任意設成 1/ln2
  D / |theta_real|     : leverage effect 的真實值僅 -0.03，分母近零會爆炸（§23）
"""
import numpy as np

from . import config as C


# ===========================================================
# Stationary block bootstrap (Politis & Romano 1994)
# ===========================================================
def block_length(T):
    return max(2, int(round(T ** C.BLOCK_LEN_EXPONENT)))


def stationary_bootstrap_idx(T, L, rng):
    """回傳長度 T 的索引陣列；block 長度 ~ Geometric(1/L)，環狀接續。

    保留時序相依（波動聚集、leverage），這是 i.i.d. bootstrap 做不到的。
    """
    idx = np.empty(T, dtype=np.int64)
    i = 0
    p = 1.0 / L
    while i < T:
        s = rng.integers(0, T)
        l = min(rng.geometric(p), T - i)
        idx[i:i + l] = (s + np.arange(l)) % T
        i += l
    return idx


def tau_univariate(x_real, metric_fn, dist_fn, B=None, rng=None, T_target=None):
    """單一序列的 tau：兩組獨立重抽之間的期望距離。

    metric_fn: 1-D array -> 標量或向量（曲線）
    dist_fn  : (theta_a, theta_b) -> 標量距離
    """
    B = C.BOOTSTRAP_B if B is None else B
    rng = rng or np.random.default_rng(C.BOOTSTRAP_SEED)
    x = np.asarray(x_real, dtype=np.float64)
    T = len(x)
    L = block_length(T)
    ds = []
    for _ in range(B):
        a = metric_fn(x[stationary_bootstrap_idx(T, L, rng)])
        b = metric_fn(x[stationary_bootstrap_idx(T, L, rng)])
        d = dist_fn(a, b)
        if np.isfinite(d):
            ds.append(d)
    return float(np.mean(ds)) if ds else np.nan


def tau_multivariate(X_real, metric_fn, dist_fn, B=None, rng=None):
    """多變量版：整列一起重抽以保留橫斷面相依（絕不可逐檔獨立重抽）。"""
    B = C.BOOTSTRAP_B if B is None else B
    rng = rng or np.random.default_rng(C.BOOTSTRAP_SEED)
    X = np.asarray(X_real, dtype=np.float64)
    T = X.shape[0]
    L = block_length(T)
    ds = []
    for _ in range(B):
        a = metric_fn(X[stationary_bootstrap_idx(T, L, rng)])
        b = metric_fn(X[stationary_bootstrap_idx(T, L, rng)])
        d = dist_fn(a, b)
        if np.isfinite(d):
            ds.append(d)
    return float(np.mean(ds)) if ds else np.nan


def bootstrap_ci_univariate(x, metric_fn, B=None, rng=None, alpha=0.05):
    """單一序列 metric 的 percentile bootstrap CI 與 SE。"""
    B = C.BOOTSTRAP_CI_B if B is None else B
    rng = rng or np.random.default_rng(C.BOOTSTRAP_SEED + 1)
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    L = block_length(T)
    vals = []
    for _ in range(B):
        v = metric_fn(x[stationary_bootstrap_idx(T, L, rng)])
        v = np.nanmean(v) if np.ndim(v) else v
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan, np.nan
    vals = np.array(vals)
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))),
            float(vals.std(ddof=1)))


# Hill 指數的收斂率：k = ceil(n^(2/3)) 使 SE(alpha_hat) ~ alpha/sqrt(k) ~ n^(-1/3)，
# 因此 tau(m) ~ m^HILL_TAU_EXPONENT。若實證擬合可靠則用擬合值，否則用此理論值。
HILL_TAU_EXPONENT = -1.0 / 3.0
HILL_TAU_EXPONENT_BOUNDS = (-0.65, -0.20)      # 擬合值超出此範圍視為不可靠，退回理論值


def tau_hill_subsampling(x, metric_fn, dist_fn, B=None, rng=None, return_detail=False):
    """Hill 指數專用的 tau —— 用**不重疊**的 m-out-of-n 子樣本，再依收斂率外推到 n。

    為什麼不能直接用 tau_univariate 的 block bootstrap：Hill 是極值順序統計量，
    一般 bootstrap 對它**不一致**（Athreya 1987）—— 重抽會重複抽到同一批極端觀測，
    低估尾部的真實變異，使 tau 偏大、分數偏鬆。實測在 primary window 上高估約 1.27 倍。

    兩個實作要點（缺一結果就會錯）：
      1) 兩個子樣本必須**不重疊**。若允許重疊，m 大時兩組共用大部分觀測，
         tau 會被人為壓向 0。
      2) m 必須滿足 2m << n。否則兩段不重疊子樣本的位置幾乎唯一，
         等於只比較了一組固定切分，不是在對 B 次抽樣取平均。
    """
    B = C.BOOTSTRAP_B if B is None else B
    rng = rng or np.random.default_rng(C.BOOTSTRAP_SEED + 2)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # m 的網格。下界 max(80, 0.08n)：m 太小時 Hill 只剩約 10 個尾部觀測，
    # 雜訊會蓋過 tau(m) 的趨勢、把擬合斜率帶到不合理的區間。
    # 上界 0.35n：確保 2m << n，兩段不重疊子樣本才有足夠的位置自由度。
    m_lo, m_hi = max(80, int(0.08 * n)), int(0.35 * n)
    if m_hi < m_lo:
        return np.nan, dict(reason=f"n={n} 太短，無法建立 subsampling 網格")
    ms = np.unique(np.round(np.geomspace(m_lo, m_hi, 4)).astype(int))
    can_fit = m_hi >= 2 * m_lo      # 網格跨度不足 2 倍時，斜率擬合不可信

    taus = []
    for m in ms:
        ds = []
        for _ in range(B):
            s1 = rng.integers(0, n - 2 * m)
            s2 = rng.integers(s1 + m, n - m)          # 強制不重疊
            d = dist_fn(metric_fn(x[s1:s1 + m]), metric_fn(x[s2:s2 + m]))
            if np.isfinite(d):
                ds.append(d)
        taus.append(np.mean(ds) if len(ds) > 0.5 * B else np.nan)

    ok = np.isfinite(taus) & (np.array(taus) > 0)
    detail = dict(ms=ms.tolist(), taus=[float(t) for t in taus])
    if can_fit and ok.sum() >= 3:
        slope, intercept = np.polyfit(np.log(ms[ok]), np.log(np.array(taus)[ok]), 1)
        lo_b, hi_b = HILL_TAU_EXPONENT_BOUNDS
        if lo_b <= slope <= hi_b:
            detail.update(exponent=float(slope), exponent_source="fitted")
        else:   # 擬合值離譜（樣本太少或雜訊過大）-> 退回理論收斂率
            slope = HILL_TAU_EXPONENT
            intercept = np.mean(np.log(np.array(taus)[ok]) - slope * np.log(ms[ok]))
            detail.update(exponent=float(slope), exponent_source="theoretical (fit out of range)")
    else:
        i = int(np.nanargmax(np.where(ok, taus, np.nan)))
        slope = HILL_TAU_EXPONENT
        intercept = np.log(taus[i]) - slope * np.log(ms[i])
        detail.update(exponent=float(slope), exponent_source="theoretical (too few usable m)")

    tau_n = float(np.exp(intercept + slope * np.log(n)))
    detail["tau_n"] = tau_n
    return (tau_n, detail) if return_detail else (tau_n, detail)


def subsample_ci_hill(x, metric_fn, B=None, rng=None, alpha=0.05, m_frac=0.5):
    """Hill 指數的 m-out-of-n subsampling 信賴區間（用於 real-metric CI 表）。

    縮放指數與 tau_hill_subsampling 一致（n^(-1/3)，來自 k = n^(2/3) 規則），
    而非一般的 sqrt(m/n) —— 後者假設 n^(-1/2) 收斂率，對此 Hill 設定並不成立。
    """
    B = C.BOOTSTRAP_CI_B if B is None else B
    rng = rng or np.random.default_rng(C.BOOTSTRAP_SEED + 2)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = min(max(60, int(0.3 * n)), n - 1)
    full = metric_fn(x)
    vals = []
    for _ in range(B):
        s = rng.integers(0, n - m)
        v = metric_fn(x[s:s + m])
        if np.isfinite(v):
            vals.append(v)
    if not vals or not np.isfinite(full):
        return np.nan, np.nan, np.nan
    vals = np.array(vals)
    scale = (m / n) ** (-HILL_TAU_EXPONENT)          # (m/n)^(1/3)
    lo = full - (np.percentile(vals, 100 * (1 - alpha / 2)) - full) * scale
    hi = full - (np.percentile(vals, 100 * alpha / 2) - full) * scale
    return float(lo), float(hi), float(vals.std(ddof=1) * scale)


# ===========================================================
# Distance -> Score
# ===========================================================
def score(D, tau, kernel=None):
    """把 raw distance 轉成 [0,1] fidelity score。"""
    kernel = C.SCORE_KERNEL if kernel is None else kernel
    if not np.isfinite(D) or not np.isfinite(tau) or tau <= 0:
        return np.nan
    z = D / tau
    if kernel == "halving":
        return float(2.0 ** (-z))
    if kernel == "gaussian":
        return float(np.exp(-0.5 * z * z))
    raise ValueError(kernel)


def signal_to_noise(theta_real, tau):
    """|theta_real| / tau。< 1 代表真實訊號本身淹沒在估計噪音裡，
    該 metric 在該 window 下統計功效不足，表格必須標記。"""
    if not np.isfinite(theta_real) or not np.isfinite(tau) or tau <= 0:
        return np.nan
    return float(abs(theta_real) / tau)
