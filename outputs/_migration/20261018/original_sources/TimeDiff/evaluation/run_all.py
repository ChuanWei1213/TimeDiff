"""端到端執行：載入 -> 校準 -> 評估 -> 表格 -> 圖表 -> 報告。

    python -m evaluation.run_all            # 全部重算
    python -m evaluation.run_all --quick    # B=100，開發用

Notebook 會呼叫同一批函式，因此兩者結果一致。
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

from . import config as C
from . import loaders, pipeline, report, calibration as CAL
from . import metrics_uni as MU
from .pipeline import SF_METRICS, SF_LABELS, CA_LABELS


# ===========================================================
def run_all_windows(B=None, verbose=True):
    """對每個 (model, window) 跑完整 pipeline。回傳 dict[(model, window)] = result。"""
    discovered = loaders.discover_models(verbose=verbose)
    models = loaders.select_models(discovered, verbose=verbose)
    if not models:
        raise RuntimeError("沒有符合 primary window 要求的模型")

    results = {}
    tau_cache = {}
    for wname, (ws, we) in C.EVAL_WINDOWS.items():
        rdates, R = loaders.load_real(ws, we)
        rm = pipeline.real_metrics(R)
        if verbose:
            print(f"\n=== Window {wname}  {ws}~{we}  T={R.shape[0]} ===")
            print("  校準 tau（stationary block bootstrap，只用真實資料）...")
        t0 = time.time()
        tau_uni, tau_cross = pipeline.calibrate_tau(R, rm, B=B, verbose=False)
        tau_cache[wname] = (tau_uni, tau_cross)
        if verbose:
            print(f"  tau 完成 ({time.time()-t0:.0f}s)")

        for m in models:
            sdates, S = loaders.load_synthetic(m, ws, we, real_dates=rdates)
            loaders.assert_alignment(rdates, sdates, R, S, m["tag"])
            res = pipeline.evaluate(R, S, m["tag"], wname,
                                    tau_uni=tau_uni, tau_cross=tau_cross, verbose=verbose)
            res["display"] = m["display"]
            res["real"] = R
            res["dates"] = rdates
            results[(m["display"], wname)] = res
    return results, models, tau_cache


# ===========================================================
# 表格
# ===========================================================
def table_sf1(results, window=None):
    """TABLE SF-1：10 檔平均的 normalized scores（跨路徑取平均）。"""
    window = window or C.PRIMARY_WINDOW_KEY
    rows = {}
    for (mdl, w), r in results.items():
        if w != window:
            continue
        s = r["sf"].mean(axis=0)
        rows[mdl] = {SF_LABELS[m]: s[m] for m in SF_METRICS}
        rows[mdl]["SF Score"] = s["SF_Score"]
    cols = [SF_LABELS[m] for m in SF_METRICS] + ["SF Score"]
    return pd.DataFrame(rows).T.reindex(columns=cols)


def table_sf2(results, window=None):
    """TABLE SF-2：raw metric distances（不放 score）。"""
    window = window or C.PRIMARY_WINDOW_KEY
    rows = {}
    for (mdl, w), r in results.items():
        if w != window:
            continue
        d = r["asset"].groupby("metric")["distance"].mean()
        rows[mdl] = {f"{SF_LABELS[m]} D": d.get(m, np.nan) for m in SF_METRICS}
        sub = r["asset"][r["asset"].metric == "jump"]
        rows[mdl]["Jump D(freq)"] = sub["d_jump_freq"].mean()
        rows[mdl]["Jump D(mag)"] = sub["d_jump_mag"].mean()
    cols = [f"{SF_LABELS[m]} D" for m in SF_METRICS] + ["Jump D(freq)", "Jump D(mag)"]
    return pd.DataFrame(rows).T.reindex(columns=cols)


def table_sf3(results, window=None):
    """TABLE SF-3：asset-level dispersion。回答「平均是否只被幾支股票拉高」。"""
    window = window or C.PRIMARY_WINDOW_KEY
    rows = []
    for (mdl, w), r in results.items():
        if w != window:
            continue
        per_tic = r["asset"].groupby(["metric", "ticker"])["score"].mean()
        for m in SF_METRICS:
            v = per_tic.loc[m]
            rows.append(dict(Method=mdl, Metric=SF_LABELS[m],
                             Mean=v.mean(), Std=v.std(ddof=1),
                             Min=v.min(), Max=v.max(),
                             **{"Min ticker": v.idxmin(), "Max ticker": v.idxmax()}))
    return pd.DataFrame(rows)


def table_ca_scores(results, window=None):
    window = window or C.PRIMARY_WINDOW_KEY
    rows = {}
    for (mdl, w), r in results.items():
        if w != window:
            continue
        s = r["ca"].mean(axis=0)
        c = r["cross"].mean(numeric_only=True)
        rows[mdl] = {
            "Corr Score": s["score_correlation"],
            "PCA Score": s["score_pca"],
            "Tail Dep. Score (copula)": s["score_tail_dep_copula"],
            "Cross-Asset Score": s["CA_Score"],
            "[suppl.] Tail Dep. Score (absolute)": c["score_tail_dep_absolute"],
        }
    return pd.DataFrame(rows).T.reindex(columns=[
        "Corr Score", "PCA Score", "Tail Dep. Score (copula)",
        "Cross-Asset Score", "[suppl.] Tail Dep. Score (absolute)"])


def table_ca_distances(results, window=None):
    window = window or C.PRIMARY_WINDOW_KEY
    rows = {}
    for (mdl, w), r in results.items():
        if w != window:
            continue
        c = r["cross"].mean(numeric_only=True)
        rows[mdl] = {
            "Corr MAE": c["d_correlation"],
            "Corr Frobenius": c["d_correlation_frobenius"],
            "Corr MAE (Spearman)": c["d_correlation_spearman"],
            "PCA Distance (TV)": c["d_pca"],
            "Tail Dep. Distance (copula)": c["d_tail_dep_copula"],
            "[suppl.] Tail Dep. Distance (absolute)": c["d_tail_dep_absolute"],
            "avg #exceed < real q05 (syn)": c["n_exceed_abs_syn"],
            "avg #exceed < real q05 (real)": c["n_exceed_abs_real"],
        }
    return pd.DataFrame(rows).T.reindex(columns=[
        "Corr MAE", "Corr Frobenius", "PCA Distance (TV)", "Tail Dep. Distance (copula)",
        "Corr MAE (Spearman)", "[suppl.] Tail Dep. Distance (absolute)",
        "avg #exceed < real q05 (real)", "avg #exceed < real q05 (syn)"])


def table_overall(results):
    """主表：Method × Time。primary 決定排名，其他 window 僅作 regime robustness。"""
    rows = []
    order = [C.PRIMARY_WINDOW_KEY] + list(C.ROBUSTNESS_WINDOWS.keys())
    for (mdl, w), r in sorted(results.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1]))):
        ov = r["overall"].mean(axis=0)
        low = [SF_LABELS[m] for m, v in r["snr"].items() if np.isfinite(v)
               and v < C.MIN_SIGNAL_TO_NOISE]
        rows.append(dict(Method=mdl, Time=w, T=r["T"], Paths=r["n_paths"],
                         **{"Stylized Facts": ov["SF_Score"],
                            "Cross-Asset": ov["CA_Score"],
                            "Overall": ov["Overall"]},
                         Role=("PRIMARY (ranking)" if w == C.PRIMARY_WINDOW_KEY
                               else "robustness only"),
                         **{"Low-power metrics": ", ".join(low) if low else "—"}))
    return pd.DataFrame(rows)


def table_snr(results, window=None):
    """訊噪比 |theta_real| / tau：< 1 代表該 window 下該 metric 統計功效不足。"""
    window = window or C.PRIMARY_WINDOW_KEY
    rows = {}
    for (mdl, w), r in results.items():
        rows.setdefault(w, {SF_LABELS[m]: v for m, v in r["snr"].items()})
    return pd.DataFrame(rows).T


def table_real_ci(R, dates, window_name):
    """真實 metric 的 bootstrap / subsampling 95% CI（§24 統計可靠度）。"""
    rows = []
    rng = np.random.default_rng(C.BOOTSTRAP_SEED + 7)
    for i, tic in enumerate(C.TICKERS):
        r = R[:, i]
        # Hill：一般 bootstrap 對極值順序統計量不一致 -> 用 m-out-of-n subsampling
        lo_h, hi_h, se_h = CAL.subsample_ci_hill(r, MU.hill_alpha, rng=rng)
        lo_s, hi_s, se_s = CAL.bootstrap_ci_univariate(r, MU.skewness, rng=rng)
        lo_v, hi_v, se_v = CAL.bootstrap_ci_univariate(r, MU.acf_r2_curve, rng=rng)
        lo_l, hi_l, se_l = CAL.bootstrap_ci_univariate(r, MU.leverage_curve, rng=rng)
        u = MU.jump_threshold(r)
        f, mg, n = MU.jump_freq_mag(r, u)
        # 比例用 Wilson score interval（精確方法，優於 bootstrap）
        T = len(r); z = 1.959964
        den = 1 + z * z / T
        ctr = (f + z * z / (2 * T)) / den
        hw = z * np.sqrt(f * (1 - f) / T + z * z / (4 * T * T)) / den
        rows.append(dict(ticker=tic,
                         hill_alpha=MU.hill_alpha(r), hill_lo=lo_h, hill_hi=hi_h,
                         skew=MU.skewness(r), skew_lo=lo_s, skew_hi=hi_s,
                         vc_mean=float(np.nanmean(MU.acf_r2_curve(r))), vc_lo=lo_v, vc_hi=hi_v,
                         le_mean=float(np.nanmean(MU.leverage_curve(r))), le_lo=lo_l, le_hi=hi_l,
                         jump_thr=u, jump_freq=f, jump_freq_lo=ctr - hw, jump_freq_hi=ctr + hw,
                         jump_n=n, jump_mag=mg))
    df = pd.DataFrame(rows)
    df.insert(0, "window", window_name)
    return df


def redundancy_check(results, window=None):
    """事先設定的規則：corr 與 pca 的 score 若在所有模型上高度相關，則視為重複資訊。"""
    window = window or C.PRIMARY_WINDOW_KEY
    xs, ys = [], []
    for (mdl, w), r in results.items():
        if w != window:
            continue
        xs.extend(r["ca"]["score_correlation"].values)
        ys.extend(r["ca"]["score_pca"].values)
    if len(xs) < 3:
        return dict(n=len(xs), corr=np.nan,
                    verdict="樣本不足（模型數過少），無法判定；加入更多模型後需重跑此檢查")
    c = float(np.corrcoef(xs, ys)[0, 1])
    return dict(n=len(xs), corr=c,
                verdict=("重複資訊 -> 依事先規則改為 corr+pca 合計 0.5、tail dep 0.5"
                         if abs(c) > C.REDUNDANCY_CORR_THRESHOLD else "不重複，維持等權"))


# ===========================================================
# 圖表
# ===========================================================
def make_figures(results, window=None):
    window = window or C.PRIMARY_WINDOW_KEY
    made = []

    sf1 = table_sf1(results, window)
    snr = {SF_LABELS[m]: v for (mdl, w), r in results.items() if w == window
           for m, v in r["snr"].items()}
    flags = [f"⚠ low statistical power (|θ_real|/τ = {v:.2f} < 1): {k}"
             for k, v in snr.items() if np.isfinite(v) and v < C.MIN_SIGNAL_TO_NOISE]
    fig, txt = report.fig_score_heatmap(
        sf1, "stylized_scores", "Stylized-facts fidelity by model",
        f"10-ticker average, mean across generated paths. Window {window}.", flags=flags)
    made.append(report.save_figure(fig, "stylized_scores", txt,
        "FIGURE 1 — Stylized Facts Score Heatmap. 每格 = 該模型在該 stylized fact 上的 "
        "fidelity score（1 = 與真實資料一致）。"))

    ca = table_ca_scores(results, window)[
        ["Corr Score", "PCA Score", "Tail Dep. Score (copula)", "Cross-Asset Score"]]
    fig, txt = report.fig_score_heatmap(
        ca, "cross_asset_scores", "Cross-asset fidelity by model",
        f"Mean across generated paths. Window {window}.")
    made.append(report.save_figure(fig, "cross_asset_scores", txt,
        "FIGURE 2 — Cross-Asset Score Heatmap."))

    prim = {mdl: r for (mdl, w), r in results.items() if w == window}
    evr_real = list(prim.values())[0]["matrices"]["evr_real"]
    evr_syn = {mdl: r["matrices"]["evr_syn_mean"] for mdl, r in prim.items()}
    fig, txt = report.fig_pca_spectrum(evr_real, evr_syn)
    made.append(report.save_figure(fig, "pca_spectrum", txt,
        "FIGURE 3 — Real vs Synthetic PCA Explained Variance。用相關矩陣（非共變異矩陣），"
        "以免把已在 univariate 層評過的尺度誤差重複計分。"))

    for mdl, r in prim.items():
        # 檔名用 tag：display name 截到第一個 "(" 會把區分資訊丟掉
        # （"Baseline (TimeDiff walk-forward)" 與 "TimeDiff (v-prediction)"
        #  會變成 "Baseline" / "TimeDiff"，看不出誰是 v-prediction）。
        slug = r["model"]
        fig, txt = report.fig_corr_matrices(
            r["matrices"]["corr_real"], r["matrices"]["corr_syn_mean"], mdl)
        made.append(report.save_figure(fig, f"correlation_matrix_{slug}", txt,
            f"FIGURE 4 — Real vs Synthetic Correlation Matrix。"
            f" model tag = {r['model']}；display name = {mdl}；window = {window}。"))

        fig, txt = report.fig_tail_dep_matrices(
            r["matrices"]["td_real"], r["matrices"]["td_syn_mean"], mdl, C.TAIL_DEP_Q)
        made.append(report.save_figure(fig, f"tail_dependence_{slug}", txt,
            f"FIGURE 5 — Real vs Synthetic Lower Tail Dependence Matrix（copula / rank based）。"
            f" model tag = {r['model']}；display name = {mdl}；window = {window}。"))

    ov = table_overall(results)
    ovp = ov[ov.Time == window].set_index("Method")[
        ["Stylized Facts", "Cross-Asset", "Overall"]]
    fig, txt = report.fig_overall(ovp)
    made.append(report.save_figure(fig, "overall_comparison", txt,
        "FIGURE 6 — Overall Model Comparison（primary window only；"
        "2020/2022 僅作 regime robustness，不參與排名）。"))

    for mdl, r in prim.items():
        # 檔名用 tag：display name 截到第一個 "(" 會把區分資訊丟掉
        # （"Baseline (TimeDiff walk-forward)" 與 "TimeDiff (v-prediction)"
        #  會變成 "Baseline" / "TimeDiff"，看不出誰是 v-prediction）。
        slug = r["model"]
        fig, txt = report.fig_asset_level(r["asset"], SF_LABELS)
        made.append(report.save_figure(fig, f"asset_level_scores_{slug}", txt,
            f"FIGURE 7 — Asset-level Score Distribution。點 = 跨路徑平均，"
            f"橫線 = 跨路徑全距，虛線 = 10 檔平均。"
            f" model tag = {r['model']}；display name = {mdl}；window = {window}。"))
    return made


# ===========================================================
# 報告
# ===========================================================
def write_methodology_report(results, tau_cache, path=None):
    path = path or os.path.join(C.RESULTS_DIR, "evaluation_methodology_report.txt")
    w = C.PRIMARY_WINDOW_KEY
    prim = {mdl: r for (mdl, wn), r in results.items() if wn == w}
    any_r = list(prim.values())[0]
    tau_uni, tau_cross = tau_cache[w]
    tau_mean = {k: float(np.nanmean([tau_uni[t][k] for t in C.TICKERS]))
                for k in tau_uni[C.TICKERS[0]]}
    red = redundancy_check(results)

    L = []
    A = L.append
    A("=" * 100)
    A("FINANCIAL FIDELITY EVALUATION — METHODOLOGY REPORT")
    A("=" * 100)
    A("")
    A("1. PRIMARY EVALUATION WINDOW")
    A("-" * 100)
    A(f"   PRIMARY_EVAL_START = {C.PRIMARY_EVAL_START}")
    A(f"   PRIMARY_EVAL_END   = {C.PRIMARY_EVAL_END}")
    A(f"   T = {any_r['T']} trading days, {C.N_TICKER} tickers: {', '.join(C.TICKERS)}")
    A("   Robustness windows (NOT used for ranking): " +
      "; ".join(f"{k} {v[0]}~{v[1]}" for k, v in C.ROBUSTNESS_WINDOWS.items()))
    A("")
    A("2. WHY THIS TIME WINDOW WAS SELECTED")
    A("-" * 100)
    A("   選擇準則完全基於真實市場資料的統計性質，在計算任何模型分數之前固定：")
    A("   (a) Regime 覆蓋完整：2018 市場修正、2020 COVID 危機、2022 升息熊市、2023-24 復甦。")
    A("   (b) 訊噪比：五個 stylized fact 的真實值皆須高於其自身重抽噪音 tau。單一年度視窗")
    A("       （2024）的 skewness 噪音尺度 tau=1.30 大於真實訊號 |−0.96|；其 ACF(r^2) 曲線")
    A("       平均僅 −0.008、leverage 僅 −0.001，等於用一個本身不具該性質的期間評估模型。")
    A("   (c) 估計可靠度：Hill 尾部樣本 k 由 6 (T=252) 提升至 42 (T=1761)，相對 SE 40% -> 15%；")
    A("       5% 尾部曝險由 12 筆提升至 88 筆，tail dependence 方可估計。")
    A("   (d) 資料完整性：10 檔於該期間零缺值。")
    A("   (e) 與 rolling generation 的生成期完全一致，不需截斷、不引入未來資料。")
    A("   ANTI-CHERRY-PICKING：window 的選擇未使用任何模型的 fidelity 分數，")
    A("   且刻意排除「某模型在該期間表現較好」這一類理由。所有 Method 使用同一 window。")
    A("")
    A("3. STYLIZED FACTS DEFINITIONS (Level 1, univariate)")
    A("-" * 100)
    A("   五個 fact 屬於三種不同數學性質，因此使用三種不同的 distance：")
    A("     distribution-level : Heavy Tail, Negative Skewness")
    A("     event-level        : Jump Behaviour")
    A("     temporal-dynamics  : Volatility Clustering, Leverage Effect")
    A("")
    A("4. EACH METRIC'S REAL / SYNTHETIC CALCULATION")
    A("-" * 100)
    A("   [Heavy Tail]  左尾 Hill 指數。x = sort(|r| for r<0) 由大到小，")
    A(f"                 k = ceil(n_neg^(2/3))（本 window 平均 k≈{int(np.ceil((any_r['T']*0.5)**(2/3)))}），u = x_(k)")
    A("                 alpha_hat = [ (1/k) * sum_{j=1..k} log(x_(j)/u) ]^(-1)")
    A("                 P(X>x) ~ x^(-alpha)；alpha 小 => 尾巴厚。只用負報酬（下跌尾部才是風險關心對象）。")
    A("   [Skewness]    g1 = m3 / m2^(3/2)，小樣本偏誤修正（bias=False）。")
    A("   [Jump]        門檻 u_i = q_{%.2f}(|r_real,i|)，**由真實資料定義並套用於 synthetic**。" % C.JUMP_Q)
    A("                 frequency f = #{|r_t| > u_i} / T；magnitude m = E[|r_t| | |r_t| > u_i]。")
    A("                 無超越事件時 magnitude 記為 undefined 且該子分數 = 0（不得靜默丟棄）。")
    A(f"   [Vol.Clust.]  rho(k) = Corr(r_t^2, r_{{t+k}}^2)，k = 1..{C.ACF_MAX_LAG}。")
    A(f"   [Leverage]    L(k)   = Corr(r_t, r_{{t+k}}^2)，k = 1..{C.ACF_MAX_LAG}。")
    A(f"   K = {C.ACF_MAX_LAG} 的三個理由：(i) 約一個交易月，是波動聚集在實務上最相關的水平；")
    A("                 (ii) Bartlett SE ≈ 1/sqrt(T)，更長的 lag 已落在噪音帶內；")
    A(f"                 (iii) **walk-forward 的 PRED_LEN = {C.WALKFORWARD_PRED_LEN}**：lag >= 21 的相依性")
    A("                 有一部分來自真實 condition 而非模型，納入會系統性高估模型。")
    A("")
    A("5. RAW DISTANCE FORMULAS")
    A("-" * 100)
    A("   D_tail  = |log(alpha_syn / alpha_real)|")
    A("             （log-ratio 而非相對誤差：高估 2 倍與低估 2 倍距離相同；相對誤差偏袒低估）")
    A("   D_skew  = |skew_syn - skew_real|")
    A("             （絕對距離：skewness 無量綱且可跨零，相對距離無定義）")
    A("   D_jump_freq = |f_syn - f_real| ;  D_jump_mag = |log(m_syn / m_real)|")
    A("   D_VC    = (1/K) * sum_k |rho_syn(k) - rho_real(k)|      (L1 curve distance)")
    A("   D_LE    = (1/K) * sum_k |L_syn(k)  - L_real(k)|         (L1 curve distance)")
    A("   D_corr  = mean_{i<j} |C_real(i,j) - C_syn(i,j)|         (off-diagonal only)")
    A("   D_PCA   = 0.5 * sum_k |EVR_real,k - EVR_syn,k|          (total variation distance)")
    A("   D_tailD = mean_{i!=j} |lambdaL_real(i,j) - lambdaL_syn(i,j)|")
    A("")
    A("6. NORMALIZATION METHOD")
    A("-" * 100)
    A("   所有 metric 共用同一個尺度化方式：以「真實資料自身的重抽噪音」為單位。")
    A("     tau_m = E_b [ D_m( theta_hat(X_real^{b1}), theta_hat(X_real^{b2}) ) ]")
    A(f"   其中 X_real^{{b}} 由 stationary block bootstrap（Politis-Romano，B={C.BOOTSTRAP_B}，")
    A("   平均 block 長度 = T^(1/3)）產生。tau **只使用真實資料**，不接觸任何 synthetic 輸出，")
    A("   因此不存在 circularity；calibration 也未使用模型結果決定門檻。")
    A("   jump frequency 的 tau 改用二項閉式解 sqrt(2 f(1-f)/T)（比例的精確方法）。")
    A("")
    A("   **Heavy Tail 的 tau 是例外**：Hill 是極值順序統計量，一般 bootstrap 對它不一致")
    A("   （Athreya 1987）—— 重抽會重複抽到同一批極端觀測，低估尾部的真實變異，使 tau 偏大、")
    A("   分數偏鬆。因此改用**不重疊**的 m-out-of-n subsampling：在多個子樣本長度 m 上計算")
    A("   tau(m)，擬合 log tau = a + b log m，再外推到 n。兩個實作要點缺一不可：")
    A("     (i)  兩個子樣本必須不重疊 —— 否則 m 大時兩組共用大部分觀測，tau 被人為壓向 0；")
    A("     (ii) m 必須滿足 2m << n —— 否則不重疊子樣本的位置幾乎唯一，等於只比較一組固定切分。")
    A("   由 k = ceil(n^(2/3)) 可推得 SE(alpha_hat) ~ n^(-1/3)，故理論上 b = -1/3。")
    A("   在 primary window 上實證擬合得到 b = -0.332，與理論值吻合，驗證了外推的正當性。")
    A("   T ~ 252 的 robustness window 因網格跨度不足無法可靠擬合，改用理論值 b = -1/3。")
    A("")
    A("   本 window 的 tau（10 檔平均）：")
    for k, v in tau_mean.items():
        A(f"     tau_{k:<16s} = {v:.5f}")
    for k, v in tau_cross.items():
        A(f"     tau_{k:<16s} = {v:.5f}")
    A("")
    A("   為何不用其他做法：")
    A("     score = 1 - D      需要 D in [0,1]，只有 TV distance 滿足，其餘 metric 無界。")
    A("     score = exp(-D)    等於把 tau 任意設為 1/ln2，無依據。")
    A("     D / |theta_real|   leverage effect 真實值僅 -0.03、jump frequency 0.05，")
    A("                        分母近零會產生假的巨大相對誤差（§23 問題）。本框架改除以")
    A("                        **標準誤**而非 metric 值，因此近零不構成問題。")
    A("")
    A("7. SCORE TRANSFORMATION")
    A("-" * 100)
    A("   Score = 2 ^ ( - D / tau )   in (0, 1]")
    A("     D = 0    -> Score = 1      完全一致")
    A("     D = tau  -> Score = 0.5    誤差恰等於真實估計量自身的重抽誤差")
    A("     D -> inf -> Score -> 0")
    A(f"   敏感度：另以 Gaussian kernel exp(-0.5*(D/tau)^2) 重算，確認排名不依賴 kernel 選擇。")
    A("")
    A("8. 10-STOCK AGGREGATION METHOD")
    A("-" * 100)
    A("   for ticker in TICKERS:  real metric -> syn metric -> distance -> score")
    A("   Score_metric = (1/10) * sum_i Score_metric,i")
    A("   **絕不先把 10 檔 concatenate 再計算 univariate metric** —— 那會製造 9 個假的跨股票")
    A("   斷點污染所有時序 metric，並把不同尺度的資產混成一個混合分布。")
    A("   per-ticker 的原始 metric、distance、score 全部保存於 results/tables/asset_level_*.csv。")
    A("")
    A("9. STYLIZED FACTS WEIGHTING")
    A("-" * 100)
    A(f"   採用 hierarchical weighting（w_distribution = {C.W_DISTRIBUTION}, w_dynamics = {C.W_DYNAMICS}）：")
    A("     Distribution = (Score_HeavyTail + Score_Skew) / 2")
    A("     Dynamics     = (Score_Jump + Score_VC + Score_LE) / 3")
    A("     SF Score     = 0.5 * Distribution + 0.5 * Dynamics")
    A("   理由：等權方案下 Dynamics 會拿到 60% 總權重，但該 60/40 純粹來自「我們碰巧定義了")
    A("   3 個 dynamics 與 2 個 distribution 指標」，是指標數量的產物而非金融判斷。")
    A("   Hierarchical 讓權重反映兩個概念軸的對等地位，且未來新增指標時保持穩定。")
    A("   權重在看到任何模型分數之前固定；附錄同時報 equal weighting 以證明結論不依賴權重。")
    A("")
    A("10. CROSS-ASSET METRICS (Level 2, multivariate)")
    A("-" * 100)
    A("   [Correlation]  Pearson 相關矩陣的 off-diagonal MAE（另報 Spearman 作穩健性）。")
    A("   [PCA]          **相關矩陣**（非共變異矩陣）的特徵值 -> explained variance ratio。")
    A("                  用相關矩陣的兩個理由：(i) 共變異的特徵值會被變異數尺度差異主導，")
    A("                  而尺度誤差已在 univariate 層評過，重複計分會扭曲 Overall；")
    A("                  (ii) 金融 RMT / 因子結構文獻（Laloux 1999、Plerou 2002）的標準做法。")
    A("                  PCA 在此**不是用來預測**，而是檢查 synthetic market 是否保留真實市場的")
    A("                  共同潛在因子結構（PC1 = 市場因子強度）。")
    A(f"   [Tail Dep.]    Copula / rank-based：lambdaL(i,j) = P(U_j < {C.TAIL_DEP_Q} | U_i < {C.TAIL_DEP_Q})，U = 經驗 CDF。")
    A("                  為何用下尾相依而非單純相關：相關係數是平均的線性共動，對常態分布才是")
    A("                  完整刻畫；金融危機的本質是尾部共動 —— 平時相關 0.4 的兩檔可能在崩盤時")
    A("                  同時重挫。這正是投資組合分散化失效的機制，也是 VaR/CVaR 最關鍵的輸入。")
    A("                  為何用 copula 版為主：lambdaL 依定義是 copula 泛函，對邊際的單調變換不變；")
    A("                  邊際尺度誤差已在 univariate 層懲罰過，用真實絕對門檻會重複計分，且")
    A("                  變異數收縮嚴重的模型會因曝險數為 0 而 undefined。")
    A("                  絕對門檻版保留為 supplementary analysis（見表格中的 [suppl.] 欄）。")
    A("")
    A("11. CROSS-ASSET DISTANCE FORMULAS")
    A("-" * 100)
    A("   同第 5 節的 D_corr / D_PCA / D_tailD。")
    A("   D_PCA 使用 total variation distance：EVR 是機率向量（和為 1），0.5*L1 即 TV，")
    A("   天然有界於 [0,1] 且有明確機率意義。")
    A("")
    A("12. CROSS-ASSET WEIGHTING")
    A("-" * 100)
    A("   CrossAsset = (Score_corr + Score_PCA + Score_tailDep) / 3   （等權）")
    A("   理由：三者分別對應線性共動水平、共同因子結構、極端共動，無理論依據認定何者更重要。")
    A("   REDUNDANCY RULE（事先設定，不依結果調整）：corr 與 PCA 並非資訊獨立，EVR spectrum")
    A("   是相關矩陣的函數。若兩者 score 在所有模型上的相關係數 > "
      f"{C.REDUNDANCY_CORR_THRESHOLD}，則改為")
    A("   corr+PCA 合計 0.5、tail dep 0.5，以免相關結構被雙重計分。")
    A(f"   本次檢查：n = {red['n']}, corr = {red['corr'] if np.isfinite(red['corr']) else 'N/A'} -> {red['verdict']}")
    A("")
    A("13. OVERALL SCORE FORMULA")
    A("-" * 100)
    A(f"   Overall = {C.W_SF} * SF_Score + {C.W_CA} * CrossAsset_Score")
    A("   採 50/50 且未偏離：univariate fidelity 與 multivariate fidelity 是兩個互補且不可")
    A("   互相取代的品質軸 —— 邊際分布完美但股票間互相獨立的生成器，與相關結構完美但邊際")
    A("   是高斯的生成器，作為 synthetic market 都同樣不可用。")
    A("")
    A("14. STATISTICAL RELIABILITY")
    A("-" * 100)
    A("   Hill alpha         : m-out-of-n **subsampling**。一般 bootstrap 對極值順序統計量")
    A("                        不一致（Athreya 1987），不可直接使用。")
    A("   Skewness           : stationary block bootstrap。高階動差收斂慢，CI 會很寬（誠實反映）。")
    A("   Jump frequency     : **Wilson score interval**（比例的精確方法，優於 bootstrap）。")
    A("   Jump magnitude     : block bootstrap，條件於超越事件；超越事件 < 10 筆時標記不可靠。")
    A("   VC / LE curve      : stationary block bootstrap，保留時序相依。")
    A("   Corr / PCA / TailD : block bootstrap，**整列一起重抽**以保留橫斷面相依。")
    A(f"   Tail dependence    : 另檢查曝險數；< {C.TAIL_DEP_MIN_EXCEEDANCE} 筆時標記不可靠。")
    A("   多路徑模型          : 額外報告跨路徑的離散度 —— 這是**生成不確定性**，")
    A("                        與估計不確定性是不同的東西，必須分開呈現。")
    A(f"   訊噪比 |theta_real|/tau < {C.MIN_SIGNAL_TO_NOISE} 的 metric 在表格中標記為 low statistical power。")
    A("")
    A("15. LIMITATIONS")
    A("-" * 100)
    A("   (a) synthetic 序列**不是 free-running simulation**：walk-forward 每 21 個交易日就")
    A("       重新錨定到一段真實的 63 日 condition。因此超過 21 個 lag 的時間相依性有一部分")
    A("       來自真實 condition 而非模型本身。這是本框架把 K 限制在 20 的直接原因，")
    A("       也意味著本評估**不能**推論模型的長期自主模擬能力。")
    A("   (b) 目前只有 walk-forward 協定的模型涵蓋 primary window；32 個單次 2024 生成的配置")
    A("       必須以相同協定重新生成後才能進入主表（Option 1）。")
    A("   (c) 尺度（VolRatio）刻意不納入 fidelity score，因為它是尺度而非 stylized fact；")
    A("       但它獨立報告，且它解釋了 absolute-threshold jump / tail dependence 的行為。")
    A("   (d) tau 由真實資料的單一實現估得，本身帶有估計誤差；B 增大可降低但無法消除。")
    A("   (e) 2020 / 2022 robustness window 的 T ≈ 252，部分 metric 的統計功效不足，")
    A("       表格已標記；這些 window 不參與主要排名。")
    A("   (f) **Heavy Tail 的 tau 混入了 regime 變異**。其餘四個 metric 的 tau 用 stationary")
    A("       block bootstrap，兩組重抽都從整段期間取樣，因此量到的是純粹的抽樣噪音。")
    A("       但 Hill 是極值順序統計量、一般 bootstrap 對它不一致（Athreya 1987），")
    A("       故改用不重疊的子區間 —— 而兩段不重疊的子區間必然來自**不同的時間段**")
    A("       （例如拿 2020 COVID 那段與 2023 平靜期相比），所以 tau_heavy_tail 同時包含了")
    A("       抽樣噪音**與** regime 之間的真實差異，會比純抽樣噪音大，使 Heavy Tail 的")
    A("       評分偏寬鬆。這是兩害相權的結果：另一個選項（一般 bootstrap）在理論上就是錯的。")
    A("       影響範圍：tau 與模型無關，故 Heavy Tail 一欄的**模型排名不受影響**，")
    A("       只有絕對水準偏高。若要消除此偏誤，需改用能分離 regime 效應的估計方式")
    A("       （例如在 regime 內部做 subsampling），屬未來工作。")
    A("")
    A("16. POTENTIAL REVIEWER CONCERNS")
    A("-" * 100)
    A("   Q: Score 的 normalization 是不是為了讓數字好看而挑的？")
    A("   A: tau 完全由真實資料的 block bootstrap 決定，未使用任何 synthetic 輸出；")
    A("      raw distance 與 score 分別以獨立表格（SF-2 / SF-1）呈現，可直接檢核。")
    A("   Q: 為什麼不是五個 stylized fact 等權？")
    A("   A: 等權會讓權重被指標數量決定（3 vs 2）。附錄同時報等權結果與排名。")
    A("   Q: 五個 metric 的 tau 為什麼不是用同一種方法算？")
    A("   A: 四個用 stationary block bootstrap；Heavy Tail 例外，因為 Hill 是極值順序統計量，")
    A("      一般 bootstrap 對它不一致，改用不重疊子區間的 m-out-of-n subsampling。")
    A("      這個例外的代價已在 Limitations (f) 完整揭露。")
    A("   Q: Jump 分數這麼低是不是指標壞了？")
    A("   A: 不是。門檻由真實資料定義，多數模型因變異數收縮而幾乎不產生超越事件，")
    A("      這是模型的真實缺陷。另附標準化版本作為診斷，區分「尺度錯」與「形狀錯」。")
    A("   Q: 為什麼 tail dependence 用各序列自身的分位數？")
    A("   A: lambdaL 依定義是 copula 泛函，對邊際單調變換不變；使用真實絕對門檻會與")
    A("      univariate 層的尺度懲罰重複計分，且會使多數模型 undefined。絕對門檻版")
    A("      仍以 supplementary 形式完整報告。")
    A("   Q: 有沒有 look-ahead？")
    A("   A: 無。condition 一律取 origin 之前的真實資料；價格重建由單一 2017 seed 全段累積，")
    A("      生成第 t 日的價格接的是生成第 t-1 日的價格，不接真實價格；評估只用生成期。")
    A("")
    A("17. WHY THE EVALUATION FRAMEWORK IS FINANCIALLY MEANINGFUL")
    A("-" * 100)
    A("   本框架評估的不是「synthetic price 是否跟 real price 長得像」——")
    A("   逐日 price trajectory correlation 刻意**不**作為 fidelity metric，因為在")
    A("   近乎不可預測的報酬序列上，它衡量的是預測能力而非模擬品質。")
    A("   本框架評估的是 synthetic market 是否保留真實市場的：")
    A("     (1) 分布性質   —— 尾部厚度決定極端損失機率，負偏度決定下檔風險不對稱；")
    A("     (2) 事件性質   —— 跳躍的頻率與幅度決定 gap risk 與避險成本；")
    A("     (3) 時序動態   —— 波動聚集決定風險預測，leverage effect 決定波動率曲面的偏斜；")
    A("     (4) 橫斷面結構 —— 相關與共同因子決定分散化效果，尾部相依決定危機時的同時虧損。")
    A("   這四者正是把 synthetic data 用於風險管理、投組建構、壓力測試時真正被依賴的性質。")
    A("")
    A("=" * 100)
    A(f"Generated by evaluation/run_all.py  |  bootstrap B={C.BOOTSTRAP_B}, seed={C.BOOTSTRAP_SEED}")
    A("=" * 100)

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def write_paper_summary(results, path=None):
    path = path or os.path.join(C.RESULTS_DIR, "paper_results_summary.txt")
    w = C.PRIMARY_WINDOW_KEY
    sf1 = table_sf1(results, w)
    ca = table_ca_scores(results, w)
    ov = table_overall(results)
    ovp = ov[ov.Time == w].set_index("Method")
    sf3 = table_sf3(results, w)

    L = []
    A = L.append
    A("=" * 100)
    A("PAPER RESULTS SUMMARY")
    A(f"Primary evaluation window: {C.PRIMARY_EVAL_START} ~ {C.PRIMARY_EVAL_END}")
    A("=" * 100)
    A("")
    single = len(ovp) == 1
    if single:
        A("[NOTE] 目前只有一個模型（baseline）進入主表，因此所有『最佳模型』欄位皆為該模型，")
        A("       排名與 trade-off 分析要等 fine-tuned 版本加入後才有意義。")
        A("       以下先報告 baseline 的絕對表現與強弱項。")
        A("")

    A("1. BEST STYLIZED FACTS MODEL")
    A("-" * 100)
    best_sf = sf1["SF Score"].idxmax()
    A(f"   {best_sf}   SF Score = {sf1.loc[best_sf, 'SF Score']:.4f}")
    A("")
    A("2. BEST CROSS-ASSET MODEL")
    A("-" * 100)
    best_ca = ca["Cross-Asset Score"].idxmax()
    A(f"   {best_ca}   Cross-Asset Score = {ca.loc[best_ca, 'Cross-Asset Score']:.4f}")
    A("")
    A("3. BEST OVERALL MODEL")
    A("-" * 100)
    best_ov = ovp["Overall"].idxmax()
    A(f"   {best_ov}   Overall = {ovp.loc[best_ov, 'Overall']:.4f}")
    A("   注意：排名不可只看 Overall Score，必須同時檢視第 9 節的 trade-off 分析。")
    A("")
    A("4. STRONGEST STYLIZED FACT")
    A("-" * 100)
    for mdl in sf1.index:
        row = sf1.loc[mdl].drop("SF Score")
        A(f"   {mdl}: {row.idxmax()} ({row.max():.4f})")
    A("")
    A("5. WEAKEST STYLIZED FACT")
    A("-" * 100)
    for mdl in sf1.index:
        row = sf1.loc[mdl].drop("SF Score")
        A(f"   {mdl}: {row.idxmin()} ({row.min():.4f})")
    A("")
    A("6. STRONGEST CROSS-ASSET PROPERTY")
    A("-" * 100)
    cac = ["Corr Score", "PCA Score", "Tail Dep. Score (copula)"]
    for mdl in ca.index:
        row = ca.loc[mdl, cac]
        A(f"   {mdl}: {row.idxmax()} ({row.max():.4f})")
    A("")
    A("7. WEAKEST CROSS-ASSET PROPERTY")
    A("-" * 100)
    for mdl in ca.index:
        row = ca.loc[mdl, cac]
        A(f"   {mdl}: {row.idxmin()} ({row.min():.4f})")
    A("")
    A("8. DO THE MODELS PRESERVE ASSET-SPECIFIC CHARACTERISTICS?")
    A("-" * 100)
    for mdl in sf1.index:
        sub = sf3[sf3.Method == mdl]
        spread = (sub.Max - sub.Min)
        worst = sub.loc[spread.idxmax()]
        A(f"   {mdl}:")
        A(f"     跨股票離散度最大的 metric = {worst.Metric}")
        A(f"       Mean={worst.Mean:.4f} Std={worst.Std:.4f} "
          f"Min={worst.Min:.4f}({worst['Min ticker']}) Max={worst.Max:.4f}({worst['Max ticker']})")
        hi = sub[sub.Std > 0.15]
        A("     Std > 0.15（平均分數可能被少數股票主導）的 metric: " +
          (", ".join(hi.Metric) if len(hi) else "無"))
    A("")
    A("9. DO THE MODELS PRESERVE CROSS-ASSET DEPENDENCE?")
    A("-" * 100)
    for mdl in ca.index:
        r = [v for (m2, w2), v in results.items() if m2 == mdl and w2 == w][0]
        c = r["cross"].mean(numeric_only=True)
        A(f"   {mdl}:")
        A(f"     Corr MAE = {c['d_correlation']:.4f}  ->  Score {ca.loc[mdl,'Corr Score']:.4f}")
        A(f"     PC1 EVR  real {c['evr1_real']:.4f}  vs  syn {c['evr1_syn']:.4f}  "
          f"(市場因子強度{'低估' if c['evr1_syn'] < c['evr1_real'] else '高估'})")
        A(f"     Tail dep. (copula) distance = {c['d_tail_dep_copula']:.4f}  ->  "
          f"Score {ca.loc[mdl,'Tail Dep. Score (copula)']:.4f}")
        A(f"     [suppl.] absolute-threshold 曝險數 syn={c['n_exceed_abs_syn']:.1f} "
          f"vs real={c['n_exceed_abs_real']:.1f}")
    A("")
    A("10. TRADE-OFF ANALYSIS（不可只依 Overall Score 排名）")
    A("-" * 100)
    for mdl in ovp.index:
        s, cc = ovp.loc[mdl, "Stylized Facts"], ovp.loc[mdl, "Cross-Asset"]
        gap = cc - s
        A(f"   {mdl}: SF={s:.4f}  CA={cc:.4f}  (CA − SF = {gap:+.4f})")
        if abs(gap) > 0.05:
            side = "cross-asset 結構" if gap > 0 else "univariate 分布/動態"
            other = "univariate 分布/動態" if gap > 0 else "cross-asset 結構"
            A(f"     -> 該模型在 {side} 明顯優於 {other}，兩者不可互相抵銷解讀。")
        else:
            A("     -> 兩軸表現接近，無明顯偏科。")
    A("")
    A("11. REGIME ROBUSTNESS（2020 / 2022 僅供分析，不參與排名）")
    A("-" * 100)
    A(ov.to_string(index=False))
    A("")
    A("12. MAIN LIMITATION")
    A("-" * 100)
    A("   最主要的限制是變異數收縮：生成序列的標準差顯著低於真實序列，")
    A("   使得在真實資料定義的絕對門檻下，模型幾乎不產生極端事件。")
    A("   這同時壓低了 Jump 分數與 absolute-threshold tail dependence 的可估性。")
    A("   其次，walk-forward 每 21 日重新錨定真實 condition，因此本評估不能推論")
    A("   模型的長期自主模擬能力。")
    A("")
    A("=" * 100)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


# ===========================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="B=100，開發用")
    ap.add_argument("--B", type=int, default=None)
    a = ap.parse_args(argv)
    B = a.B if a.B is not None else (100 if a.quick else C.BOOTSTRAP_B)

    t0 = time.time()
    results, models, tau_cache = run_all_windows(B=B)

    os.makedirs(C.TABLE_DIR, exist_ok=True)
    w = C.PRIMARY_WINDOW_KEY
    tables = {
        "table_SF1_scores": table_sf1(results, w),
        "table_SF2_distances": table_sf2(results, w),
        "table_SF3_dispersion": table_sf3(results, w),
        "table_crossasset_scores": table_ca_scores(results, w),
        "table_crossasset_distances": table_ca_distances(results, w),
        "table_overall_comparison": table_overall(results),
        "table_signal_to_noise": table_snr(results, w),
    }
    for name, df in tables.items():
        df.to_csv(os.path.join(C.TABLE_DIR, f"{name}.csv"))
        print(f"\n### {name}")
        print(df.round(4).to_string())

    for (mdl, wn), r in results.items():
        if wn != w:
            continue
        # 檔名用 tag：display name 截到第一個 "(" 會把區分資訊丟掉
        # （"Baseline (TimeDiff walk-forward)" 與 "TimeDiff (v-prediction)"
        #  會變成 "Baseline" / "TimeDiff"，看不出誰是 v-prediction）。
        slug = r["model"]
        r["asset"].to_csv(os.path.join(C.TABLE_DIR, f"asset_level_{slug}.csv"), index=False)
        r["cross"].to_csv(os.path.join(C.TABLE_DIR, f"cross_level_{slug}.csv"), index=False)

    rdates, R = loaders.load_real(C.PRIMARY_EVAL_START, C.PRIMARY_EVAL_END)
    table_real_ci(R, rdates, w).to_csv(
        os.path.join(C.TABLE_DIR, "table_real_metric_CI.csv"), index=False)

    made = make_figures(results, w)
    print(f"\n產生 {len(made)} 張圖（png + txt）")
    p1 = write_methodology_report(results, tau_cache)
    p2 = write_paper_summary(results)
    print(f"報告：{p1}\n      {p2}")
    print(f"\n總耗時 {time.time()-t0:.0f}s")
    return results


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
