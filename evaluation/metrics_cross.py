"""Level 2 — Cross-Asset / Multivariate Structure 的 raw metrics 與 distances.

三個 metric 分別對應：
    correlation      : 線性共動的**水平**（逐元素）
    PCA / eigenvalue : 共同潛在因子**結構**（譜）
    tail dependence  : **極端**共動（copula）
"""
import numpy as np
import pandas as pd

from . import config as C


# ===========================================================
# 1. Correlation Matrix
# ===========================================================
def corr_matrix(X, method="pearson"):
    """X: (T, N) -> (N, N) 相關矩陣。"""
    X = np.asarray(X, dtype=np.float64)
    if method == "pearson":
        return np.corrcoef(X, rowvar=False)
    if method == "spearman":
        return np.corrcoef(pd.DataFrame(X).rank().values, rowvar=False)
    raise ValueError(method)


def _offdiag(M):
    """取上三角（不含對角線）的 N(N-1)/2 個元素。"""
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def d_correlation(C_syn, C_real):
    """off-diagonal MAE。對角線恆為 1，納入會人為稀釋誤差。"""
    return float(np.mean(np.abs(_offdiag(C_syn) - _offdiag(C_real))))


# ===========================================================
# 2. PCA / Eigenvalue Spectrum
# ===========================================================
def evr_spectrum(X, use_correlation=True):
    """回傳 explained variance ratio 向量 (N,)，由大到小。

    使用**相關矩陣**而非共變異矩陣，兩個理由：
      1) 共變異的特徵值會被變異數尺度差異主導。多數模型 VolRatio 僅 0.2~0.6，
         用共變異等於把已在 univariate 層評過的尺度誤差重複計分。
      2) 金融 RMT / 因子結構文獻（Laloux 1999、Plerou 2002）的標準做法即為相關矩陣。
    """
    X = np.asarray(X, dtype=np.float64)
    M = np.corrcoef(X, rowvar=False) if use_correlation else np.cov(X, rowvar=False)
    w = np.linalg.eigvalsh(M)[::-1]
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, np.nan)


def d_pca(evr_syn, evr_real):
    """Total variation distance = 0.5 * sum_k |EVR_syn(k) - EVR_real(k)|.

    EVR 是機率向量（兩者皆和為 1），其 0.5*L1 正是 TV distance，
    天然有界於 [0,1] 且有明確機率意義。
    """
    return float(0.5 * np.sum(np.abs(np.asarray(evr_syn) - np.asarray(evr_real))))


# ===========================================================
# 3. Lower Tail Dependence
# ===========================================================
def _rank_uniform(X):
    """逐欄轉成經驗 CDF 值 U = rank/(T+1)。"""
    T = X.shape[0]
    return (np.argsort(np.argsort(X, axis=0), axis=0) + 1.0) / (T + 1.0)


def tail_dep_copula(X, u=None):
    """Copula / rank-based 下尾相依矩陣 lambda_L(i,j) = P(U_j < u | U_i < u).

    這是尾部相依係數的**標準定義**：它是 copula 泛函，對邊際的單調變換不變。
    使用各序列自身的經驗 CDF 不是「挑對自己有利的門檻」，而是該量的定義；
    邊際尺度的誤差已在 univariate 層被懲罰過。
    """
    u = C.TAIL_DEP_Q if u is None else u
    X = np.asarray(X, dtype=np.float64)
    U = _rank_uniform(X)
    below = U < u
    N = X.shape[1]
    M = np.full((N, N), np.nan)
    for i in range(N):
        ni = below[:, i].sum()
        if ni == 0:
            continue
        for j in range(N):
            M[i, j] = (below[:, j] & below[:, i]).sum() / ni
    return M


def tail_dep_absolute(X, thr_real, u_nominal=None):
    """Absolute-threshold 版：門檻一律來自**真實資料**（supplementary analysis）。

    回傳 (matrix, n_exceed) — n_exceed 是每個條件事件的樣本數，
    低於 config.TAIL_DEP_MIN_EXCEEDANCE 時該模型的此指標不可靠。
    變異數收縮嚴重的模型可能完全沒有超越事件，此時矩陣為 NaN（undefined），
    **不可當成 0 混入平均**。
    """
    X = np.asarray(X, dtype=np.float64)
    below = X < np.asarray(thr_real)[np.newaxis, :]
    N = X.shape[1]
    M = np.full((N, N), np.nan)
    n_ex = below.sum(axis=0)
    for i in range(N):
        if n_ex[i] == 0:
            continue
        for j in range(N):
            M[i, j] = (below[:, j] & below[:, i]).sum() / n_ex[i]
    return M, n_ex


def absolute_tail_thresholds(X_real, q=None):
    """真實資料的下尾門檻（逐檔）。"""
    q = C.TAIL_DEP_Q if q is None else q
    return np.quantile(np.asarray(X_real, dtype=np.float64), q, axis=0)


def d_tail_dep(M_syn, M_real):
    """非對角元素的 MAE。任一方為 NaN 的位置會被排除；
    若可用位置少於一半則回傳 NaN（indicating undefined）。"""
    N = M_real.shape[0]
    mask = ~np.eye(N, dtype=bool)
    a, b = M_syn[mask], M_real[mask]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 0.5 * mask.sum():
        return np.nan
    return float(np.mean(np.abs(a[ok] - b[ok])))


def d_correlation_frobenius(C_syn, C_real):
    """|| Corr(syn) - Corr(real) ||_F —— 與訓練用的 cross-asset loss 同一個量。

    訓練端最小化的是它的平方；此處回傳未平方的 Frobenius 距離，
    數量級與 Corr MAE 可比，方便並列閱讀。對角線貢獻恆為 0。
    """
    return float(np.linalg.norm(np.asarray(C_syn) - np.asarray(C_real), ord="fro"))
