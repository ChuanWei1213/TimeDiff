"""Level 1 — Univariate Stylized Facts 的 raw metrics 與 metric-specific distances.

五個 fact 屬於三種不同的數學性質，因此使用三種不同的 distance：
    distribution-level : heavy tail (log-ratio), skewness (absolute)
    event-level        : jump      (proportion diff + log-ratio of magnitude)
    temporal-dynamics  : vol clustering / leverage (L1 curve distance)
"""
import numpy as np
from scipy import stats

from . import config as C


# ===========================================================
# 1. Heavy Tails — 左尾 Hill 指數
# ===========================================================
def hill_alpha(r, k=None, k_mult=1.0):
    """左尾 Hill 尾部指數 alpha。P(X > x) ~ x^(-alpha)，alpha 小 => 尾巴厚。

    只用負報酬（下跌尾部才是金融風險關心的對象）。
    k = ceil(n_neg ** 2/3)：k-based 門檻而非固定分位數，
    以避免小樣本時尾部觀測數塌陷（q95 在 T=252 下只剩 k≈6）。
    """
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    x = np.sort(np.abs(r[r < 0]))[::-1]        # 由大到小
    n = len(x)
    if n < 20:
        return np.nan
    if k is None:
        k = int(np.ceil(n ** C.HILL_K_EXPONENT * k_mult))
    k = int(min(max(k, 5), n - 1))
    u = x[k]                                    # 第 (k+1) 大的觀測作為門檻
    if u <= 0:
        return np.nan
    logs = np.log(x[:k] / u)
    m = logs.mean()
    return 1.0 / m if m > 0 else np.nan


def d_heavy_tail(alpha_syn, alpha_real):
    """log-ratio 距離。

    alpha 是正的尺度參數、估計量近似對數常態。log-ratio 對「高估 2 倍」與
    「低估 2 倍」給出相同距離，而相對誤差 |da|/a 會不對稱地偏袒低估。
    """
    if not (np.isfinite(alpha_syn) and np.isfinite(alpha_real)) \
            or alpha_syn <= 0 or alpha_real <= 0:
        return np.nan
    return abs(np.log(alpha_syn / alpha_real))


# ===========================================================
# 2. Negative Skewness
# ===========================================================
def skewness(r):
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 8:
        return np.nan
    return float(stats.skew(r, bias=False))     # bias=False：小樣本偏誤修正


def d_skewness(s_syn, s_real):
    """絕對距離。skewness 無量綱且可跨零，相對距離無定義。"""
    if not (np.isfinite(s_syn) and np.isfinite(s_real)):
        return np.nan
    return abs(s_syn - s_real)


def skew_sign_agreement(s_syn, s_real):
    """符號一致 = 1，否則 0。與 magnitude distance 分開呈現。"""
    if not (np.isfinite(s_syn) and np.isfinite(s_real)):
        return np.nan
    return float(np.sign(s_syn) == np.sign(s_real))


# ===========================================================
# 3. Jump Behaviour
# ===========================================================
def jump_threshold(r_real, q=None):
    """由**真實資料**定義的絕對門檻 u = q_q(|r_real|)。synthetic 一律套用此門檻。"""
    q = C.JUMP_Q if q is None else q
    r = np.asarray(r_real, dtype=np.float64)
    return float(np.quantile(np.abs(r[np.isfinite(r)]), q))


def jump_freq_mag(r, u):
    """在給定門檻 u 下的 (frequency, mean magnitude, n_exceed)。

    無超越事件時 magnitude 回傳 nan 並由呼叫端記為 'no exceedance'，
    不可靜默丟棄（那會讓最差的模型從平均中消失）。
    """
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    a = np.abs(r)
    ex = a > u
    n = int(ex.sum())
    f = n / len(a) if len(a) else np.nan
    m = float(a[ex].mean()) if n > 0 else np.nan
    return f, m, n


def d_jump_freq(f_syn, f_real):
    """比例差。tau 用二項標準誤的閉式解，不需 bootstrap。"""
    if not (np.isfinite(f_syn) and np.isfinite(f_real)):
        return np.nan
    return abs(f_syn - f_real)


def tau_jump_freq(f_real, T):
    """兩組獨立比例估計之差的標準差 = sqrt(2 f(1-f)/T)。"""
    return float(np.sqrt(2.0 * f_real * (1.0 - f_real) / T))


def d_jump_mag(m_syn, m_real):
    """正的尺度參數 → log-ratio（理由同 Hill）。"""
    if not (np.isfinite(m_syn) and np.isfinite(m_real)) or m_syn <= 0 or m_real <= 0:
        return np.nan
    return abs(np.log(m_syn / m_real))


def bns_relative_jump(r, clip=False):
    """Barndorff-Nielsen & Shephard 相對跳躍測度，**預設不裁切**。

    既有程式碼的 max(0, .) 使所有 synthetic 皆為 0（見 design doc BUG-1），
    因此這裡只作為診斷欄，回傳未裁切的原值。
    """
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return np.nan
    rv = np.sum(r ** 2)
    bv = (np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    if rv < 1e-12:
        return np.nan
    v = (rv - bv) / rv
    return max(0.0, v) if clip else v


# ===========================================================
# 4. Volatility Clustering — ACF(r^2) 曲線
# ===========================================================
def acf_r2_curve(r, K=None):
    """rho(k) = Corr(r_t^2, r_{t+k}^2)，k = 1..K。

    proxy 用 r^2（對應 realized variance，與 GARCH / BNS 文獻一致）。
    """
    K = C.ACF_MAX_LAG if K is None else K
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    v = r ** 2
    out = np.full(K, np.nan)
    if len(v) < K + 10:
        return out
    for k in range(1, K + 1):
        a, b = v[:-k], v[k:]
        sa, sb = a.std(), b.std()
        if sa > 0 and sb > 0:
            out[k - 1] = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return out


# ===========================================================
# 5. Leverage Effect — Corr(r_t, r_{t+k}^2) 曲線
# ===========================================================
def leverage_curve(r, K=None):
    """L(k) = Corr(r_t, r_{t+k}^2)，k = 1..K。真實市場通常為負。"""
    K = C.ACF_MAX_LAG if K is None else K
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    out = np.full(K, np.nan)
    if len(r) < K + 10:
        return out
    for k in range(1, K + 1):
        a, b = r[:-k], r[k:] ** 2
        sa, sb = a.std(), b.std()
        if sa > 0 and sb > 0:
            out[k - 1] = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return out


def d_curve_l1(c_syn, c_real):
    """L1 曲線距離：mean_k |c_syn(k) - c_real(k)|。

    對單一 lag 的離群值比 L2 穩健；同時檢驗曲線的水平與衰減形狀。
    """
    c_syn, c_real = np.asarray(c_syn), np.asarray(c_real)
    m = np.isfinite(c_syn) & np.isfinite(c_real)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(c_syn[m] - c_real[m])))


def curve_sign_agreement(c_syn, c_real):
    """逐 lag 符號一致率。用於 leverage effect 的方向診斷。"""
    c_syn, c_real = np.asarray(c_syn), np.asarray(c_real)
    m = np.isfinite(c_syn) & np.isfinite(c_real)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(c_syn[m]) == np.sign(c_real[m])))


# ===========================================================
# 輔助：尺度（不是 stylized fact，但必須獨立報告）
# ===========================================================
def vol_ratio(r_syn, r_real):
    """生成序列與真實序列的標準差比。1.0 = 尺度正確。

    刻意不納入 fidelity score：它是尺度而非 stylized fact。
    但必須報告，因為它解釋了 absolute-threshold jump / tail dependence 的行為。
    """
    a, b = np.std(r_syn), np.std(r_real)
    return float(a / b) if b > 0 else np.nan
