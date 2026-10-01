"""七步 pipeline：Raw Metric -> Distance -> Normalized -> Score -> Ticker Agg -> Category -> Overall.

每一步的中間結果都完整保存，raw distance 與 score 絕不混在同一張表。
"""
import numpy as np
import pandas as pd

from . import config as C
from . import metrics_uni as MU
from . import metrics_cross as MC
from . import calibration as CAL

SF_METRICS = ["heavy_tail", "skewness", "jump", "vol_clustering", "leverage"]
SF_LABELS = {
    "heavy_tail":     "Heavy Tail (Hill alpha)",
    "skewness":       "Negative Skewness",
    "jump":           "Jump Behaviour",
    "vol_clustering": "Volatility Clustering",
    "leverage":       "Leverage Effect",
}
CA_LABELS = {
    "correlation":      "Correlation Matrix",
    "pca":              "PCA / Eigenvalue Spectrum",
    "tail_dep_copula":  "Tail Dependence (copula)",
    "tail_dep_absolute": "Tail Dependence (absolute thr.)",
}


# ===========================================================
# STEP 1 — 真實資料的 raw metrics（逐檔）
# ===========================================================
def real_metrics(R):
    """R: (T, 10) -> dict of per-ticker real metrics + jump thresholds."""
    T = R.shape[0]
    out = {"T": T, "per_ticker": {}}
    for i, tic in enumerate(C.TICKERS):
        r = R[:, i]
        u = MU.jump_threshold(r)
        f, m, n = MU.jump_freq_mag(r, u)
        out["per_ticker"][tic] = dict(
            alpha=MU.hill_alpha(r),
            skew=MU.skewness(r),
            jump_u=u, jump_f=f, jump_m=m, jump_n=n,
            vc_curve=MU.acf_r2_curve(r),
            le_curve=MU.leverage_curve(r),
            bns_rj=MU.bns_relative_jump(r),
            std=float(np.std(r)),
        )
    return out


# ===========================================================
# STEP 3 — tau 校準（只用真實資料）
# ===========================================================
def calibrate_tau(R, real_m, B=None, verbose=True):
    """回傳 tau_uni[ticker][metric] 與 tau_cross[metric]。"""
    B = C.BOOTSTRAP_B if B is None else B
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    T = R.shape[0]
    tau_uni, hill_detail = {}, {}
    for i, tic in enumerate(C.TICKERS):
        r = R[:, i]
        rm = real_m["per_ticker"][tic]
        u = rm["jump_u"]
        # Hill 用 subsampling（極值順序統計量，一般 bootstrap 不一致）；
        # 其餘用 stationary block bootstrap。
        tau_ht, ht_detail = CAL.tau_hill_subsampling(r, MU.hill_alpha, MU.d_heavy_tail, B, rng)
        hill_detail[tic] = ht_detail
        tau_uni[tic] = dict(
            heavy_tail=tau_ht,
            skewness=CAL.tau_univariate(r, MU.skewness, MU.d_skewness, B, rng),
            # jump frequency 用二項閉式解（比例的精確方法，優於 bootstrap）
            jump_freq=MU.tau_jump_freq(rm["jump_f"], T),
            jump_mag=CAL.tau_univariate(
                r, lambda z, _u=u: MU.jump_freq_mag(z, _u)[1], MU.d_jump_mag, B, rng),
            vol_clustering=CAL.tau_univariate(r, MU.acf_r2_curve, MU.d_curve_l1, B, rng),
            leverage=CAL.tau_univariate(r, MU.leverage_curve, MU.d_curve_l1, B, rng),
        )
        if verbose:
            print(f"    tau[{tic}] done", end="\r")

    thr = MC.absolute_tail_thresholds(R)
    tau_cross = dict(
        correlation=CAL.tau_multivariate(R, MC.corr_matrix, MC.d_correlation, B, rng),
        pca=CAL.tau_multivariate(R, MC.evr_spectrum, MC.d_pca, B, rng),
        tail_dep_copula=CAL.tau_multivariate(R, MC.tail_dep_copula, MC.d_tail_dep, B, rng),
        tail_dep_absolute=CAL.tau_multivariate(
            R, lambda Z, _t=thr: MC.tail_dep_absolute(Z, _t)[0], MC.d_tail_dep, B, rng),
    )
    if verbose:
        src = {d.get("exponent_source") for d in hill_detail.values()}
        print(f"    tau 校準完成（Hill 用 subsampling，縮放指數來源: {', '.join(sorted(src))}）")
    return tau_uni, tau_cross


# ===========================================================
# STEP 2+4 — synthetic raw metrics -> distance -> score（逐路徑逐檔）
# ===========================================================
def evaluate_paths_univariate(R, S, real_m, tau_uni, kernel=None):
    """S: (P, T, 10)。回傳 asset-level long DataFrame（每 path×ticker×metric 一列）。"""
    rows = []
    P = S.shape[0]
    for p in range(P):
        for i, tic in enumerate(C.TICKERS):
            rs, rr = S[p, :, i], R[:, i]
            rm = real_m["per_ticker"][tic]
            tu = tau_uni[tic]

            # 1. Heavy tail
            a_s = MU.hill_alpha(rs)
            d = MU.d_heavy_tail(a_s, rm["alpha"])
            rows.append(dict(path=p, ticker=tic, metric="heavy_tail",
                             real=rm["alpha"], syn=a_s, distance=d,
                             tau=tu["heavy_tail"],
                             score=CAL.score(d, tu["heavy_tail"], kernel)))

            # 2. Skewness
            s_s = MU.skewness(rs)
            d = MU.d_skewness(s_s, rm["skew"])
            rows.append(dict(path=p, ticker=tic, metric="skewness",
                             real=rm["skew"], syn=s_s, distance=d,
                             tau=tu["skewness"],
                             score=CAL.score(d, tu["skewness"], kernel),
                             sign_agree=MU.skew_sign_agreement(s_s, rm["skew"])))

            # 3. Jump（真實門檻；frequency 與 magnitude 各自轉分數後等權平均）
            f_s, m_s, n_s = MU.jump_freq_mag(rs, rm["jump_u"])
            d_f = MU.d_jump_freq(f_s, rm["jump_f"])
            d_m = MU.d_jump_mag(m_s, rm["jump_m"])
            sc_f = CAL.score(d_f, tu["jump_freq"], kernel)
            # 無超越事件 -> magnitude 未定義 -> 該子分數記 0（不可靜默丟棄）
            sc_m = CAL.score(d_m, tu["jump_mag"], kernel) if n_s > 0 else 0.0
            rows.append(dict(path=p, ticker=tic, metric="jump",
                             real=rm["jump_f"], syn=f_s,
                             distance=d_f,          # 主要 raw distance = frequency 差
                             tau=tu["jump_freq"],
                             score=0.5 * sc_f + 0.5 * (sc_m if np.isfinite(sc_m) else 0.0),
                             jump_freq_syn=f_s, jump_mag_syn=m_s, jump_n_syn=n_s,
                             jump_freq_real=rm["jump_f"], jump_mag_real=rm["jump_m"],
                             d_jump_freq=d_f, d_jump_mag=d_m,
                             score_jump_freq=sc_f, score_jump_mag=sc_m))

            # 4. Volatility clustering
            c_s = MU.acf_r2_curve(rs)
            d = MU.d_curve_l1(c_s, rm["vc_curve"])
            rows.append(dict(path=p, ticker=tic, metric="vol_clustering",
                             real=float(np.nanmean(rm["vc_curve"])),
                             syn=float(np.nanmean(c_s)), distance=d,
                             tau=tu["vol_clustering"],
                             score=CAL.score(d, tu["vol_clustering"], kernel)))

            # 5. Leverage
            l_s = MU.leverage_curve(rs)
            d = MU.d_curve_l1(l_s, rm["le_curve"])
            rows.append(dict(path=p, ticker=tic, metric="leverage",
                             real=float(np.nanmean(rm["le_curve"])),
                             syn=float(np.nanmean(l_s)), distance=d,
                             tau=tu["leverage"],
                             score=CAL.score(d, tu["leverage"], kernel),
                             sign_agree=MU.curve_sign_agreement(l_s, rm["le_curve"]),
                             direction_negative=float(np.nanmean(l_s) < 0)))
    return pd.DataFrame(rows)


def evaluate_paths_cross(R, S, tau_cross, kernel=None):
    """Cross-asset：逐路徑一列。"""
    thr = MC.absolute_tail_thresholds(R)
    Cr_p = MC.corr_matrix(R, "pearson")
    Cr_s = MC.corr_matrix(R, "spearman")
    evr_r = MC.evr_spectrum(R)
    td_r = MC.tail_dep_copula(R)
    tda_r, n_ex_r = MC.tail_dep_absolute(R, thr)

    rows = []
    for p in range(S.shape[0]):
        X = S[p]
        Cs_p = MC.corr_matrix(X, "pearson")
        Cs_s = MC.corr_matrix(X, "spearman")
        evr_s = MC.evr_spectrum(X)
        td_s = MC.tail_dep_copula(X)
        tda_s, n_ex_s = MC.tail_dep_absolute(X, thr)

        d_corr = MC.d_correlation(Cs_p, Cr_p)
        d_corr_fro = MC.d_correlation_frobenius(Cs_p, Cr_p)
        d_corr_sp = MC.d_correlation(Cs_s, Cr_s)
        d_pca = MC.d_pca(evr_s, evr_r)
        d_td = MC.d_tail_dep(td_s, td_r)
        d_tda = MC.d_tail_dep(tda_s, tda_r)

        rows.append(dict(
            path=p,
            d_correlation=d_corr, d_correlation_frobenius=d_corr_fro,
            d_correlation_spearman=d_corr_sp,
            d_pca=d_pca, d_tail_dep_copula=d_td, d_tail_dep_absolute=d_tda,
            score_correlation=CAL.score(d_corr, tau_cross["correlation"], kernel),
            score_pca=CAL.score(d_pca, tau_cross["pca"], kernel),
            score_tail_dep_copula=CAL.score(d_td, tau_cross["tail_dep_copula"], kernel),
            score_tail_dep_absolute=CAL.score(d_tda, tau_cross["tail_dep_absolute"], kernel),
            evr1_real=float(evr_r[0]), evr1_syn=float(evr_s[0]),
            n_exceed_abs_syn=float(np.mean(n_ex_s)), n_exceed_abs_real=float(np.mean(n_ex_r)),
            vol_ratio=float(np.mean(X.std(axis=0) / R.std(axis=0))),
        ))
    matrices = dict(corr_real=Cr_p, evr_real=evr_r, td_real=td_r, tda_real=tda_r,
                    corr_syn_mean=np.mean([MC.corr_matrix(S[p]) for p in range(S.shape[0])], axis=0),
                    evr_syn_mean=np.mean([MC.evr_spectrum(S[p]) for p in range(S.shape[0])], axis=0),
                    td_syn_mean=np.nanmean([MC.tail_dep_copula(S[p]) for p in range(S.shape[0])], axis=0))
    return pd.DataFrame(rows), matrices


# ===========================================================
# STEP 5+6+7 — 聚合
# ===========================================================
def aggregate_sf(asset_df, weighting=None):
    """逐檔 score -> 10 檔平均 -> category -> SF score。每條 path 各算一次。"""
    weighting = weighting or C.SF_WEIGHTING
    # STEP 5：ticker aggregation（先逐檔算 score，再平均；絕不 concatenate）
    per_path = (asset_df.groupby(["path", "metric"])["score"].mean()
                .unstack("metric").reindex(columns=SF_METRICS))
    # STEP 6：category
    dist = per_path[C.DISTRIBUTION_METRICS].mean(axis=1)
    dyn = per_path[C.DYNAMICS_METRICS].mean(axis=1)
    per_path["Distribution"] = dist
    per_path["Dynamics"] = dyn
    if weighting == "hierarchical":
        per_path["SF_Score"] = C.W_DISTRIBUTION * dist + C.W_DYNAMICS * dyn
    else:
        per_path["SF_Score"] = per_path[SF_METRICS].mean(axis=1)
    return per_path


def aggregate_ca(cross_df, metrics=None):
    metrics = metrics or C.CROSS_ASSET_METRICS
    cols = [f"score_{m}" for m in metrics]
    out = cross_df.set_index("path")[cols].copy()
    out["CA_Score"] = out[cols].mean(axis=1)
    return out


def overall(sf_per_path, ca_per_path):
    df = pd.DataFrame({
        "SF_Score": sf_per_path["SF_Score"],
        "CA_Score": ca_per_path["CA_Score"],
    })
    df["Overall"] = C.W_SF * df["SF_Score"] + C.W_CA * df["CA_Score"]
    return df


# ===========================================================
# 單一 (model, window) 的完整評估
# ===========================================================
def evaluate(R, S, model_tag, window_name, tau_uni=None, tau_cross=None,
             B=None, kernel=None, verbose=True):
    if verbose:
        print(f"  [{model_tag} @ {window_name}] T={R.shape[0]}, paths={S.shape[0]}")
    rm = real_metrics(R)
    if tau_uni is None or tau_cross is None:
        if verbose:
            print("    校準 tau（stationary block bootstrap，只用真實資料）...")
        tau_uni, tau_cross = calibrate_tau(R, rm, B=B, verbose=verbose)

    asset_df = evaluate_paths_univariate(R, S, rm, tau_uni, kernel)
    cross_df, mats = evaluate_paths_cross(R, S, tau_cross, kernel)

    asset_df.insert(0, "window", window_name)
    asset_df.insert(0, "model", model_tag)
    cross_df.insert(0, "window", window_name)
    cross_df.insert(0, "model", model_tag)

    sf = aggregate_sf(asset_df)
    ca = aggregate_ca(cross_df)
    ov = overall(sf, ca)

    # 訊噪比：|theta_real| / tau，< 1 代表該 window 下統計功效不足
    snr = {}
    for m in SF_METRICS:
        key = {"heavy_tail": "alpha", "skewness": "skew", "jump": "jump_f"}.get(m)
        vals = []
        for tic in C.TICKERS:
            r = rm["per_ticker"][tic]
            theta = (r[key] if key else
                     float(np.nanmean(r["vc_curve" if m == "vol_clustering" else "le_curve"])))
            tkey = "jump_freq" if m == "jump" else m
            vals.append(CAL.signal_to_noise(theta, tau_uni[tic][tkey]))
        snr[m] = float(np.nanmean(vals))

    return dict(model=model_tag, window=window_name, real_metrics=rm,
                tau_uni=tau_uni, tau_cross=tau_cross,
                asset=asset_df, cross=cross_df, sf=sf, ca=ca, overall=ov,
                matrices=mats, snr=snr, T=R.shape[0], n_paths=S.shape[0])
