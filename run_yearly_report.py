"""逐年評估報表：2018–2024 每年一張表 + 綜合表。

模型只用「單層結構」的 v-prediction（dow10_wf_vpred），不含白化 / GARCH / 重尾校準，
以呈現 parameterization 本身的效果。

輸出欄位全部是 **10 檔平均的 normalized score**（0~1，越高越接近真實），
把原本分開的 SF-1 與 Cross-Asset 兩張表合併成一張。

日期對齊：dva 缺 2024-12-31，為使六個方法在完全相同的日期上比較，
該日自所有方法一併排除（1,761 天少 1 天，統計上無影響）。
"""
import os, sys, warnings
sys.path.insert(0, os.path.abspath("."))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from evaluation import config as C, loaders, pipeline
from evaluation.pipeline import SF_METRICS

OUT = "results/yearly_report"
METHODS = [("dow10_wf_v2",    "TimeDiff(Baseline)"),
           ("dow10_wf_vpred", "TimeDiff(v-prediction)"),
           ("dva",            "DVA"),
           ("cndiff",         "CN-Diff"),
           ("nsdiff",         "NS-Diff"),
           ("armd",           "ARMD")]

WINDOWS = {f"{y}": (f"{y}-01-01", f"{y}-12-30" if y == 2024 else f"{y}-12-31")
           for y in range(2018, 2025)}
WINDOWS["Overall 2018-2024"] = ("2018-01-02", "2024-12-30")

COLS = ["Heavy Tail", "Neg. Skew", "Jump", "Vol. Cluster", "Leverage",
        "Corr MAE", "PCA Distance", "Tail Dependence", "SF", "Cross", "Overall"]
SF_MAP = dict(zip(SF_METRICS, COLS[:5]))


def build(window_name, start, end, B=300):
    rdates, R = loaders.load_real(start, end)
    rm = pipeline.real_metrics(R)
    tau_uni, tau_cross = pipeline.calibrate_tau(R, rm, B=B, verbose=False)
    rows, snr = {}, None
    for tag, disp in METHODS:
        m = [x for x in loaders.discover_models(verbose=False) if x["tag"] == tag]
        if not m:
            print(f"  ⚠ {tag} 找不到，跳過"); continue
        sdates, S = loaders.load_synthetic(m[0], start, end, real_dates=rdates)
        loaders.assert_alignment(rdates, sdates, R, S, tag)
        res = pipeline.evaluate(R, S, tag, window_name,
                                tau_uni=tau_uni, tau_cross=tau_cross, verbose=False)
        sf = res["sf"].mean(axis=0)
        ca = res["ca"].mean(axis=0)
        ov = res["overall"].mean(axis=0)
        rows[disp] = {**{SF_MAP[k]: sf[k] for k in SF_METRICS},
                      "Corr MAE":        ca["score_correlation"],
                      "PCA Distance":    ca["score_pca"],
                      "Tail Dependence": ca["score_tail_dep_copula"],
                      "SF":      ov["SF_Score"],
                      "Cross":   ov["CA_Score"],
                      "Overall": ov["Overall"]}
        snr = res["snr"]
    df = pd.DataFrame(rows).T.reindex(columns=COLS)
    df = df.reindex([d for _, d in METHODS if d in df.index])
    return df, len(rdates), snr


if __name__ == "__main__":
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    allsnr = {}
    for wname, (s, e) in WINDOWS.items():
        print(f"\n{'='*100}\n### {wname}   ({s} ~ {e})")
        df, T, snr = build(wname, s, e)
        allsnr[wname] = snr
        low = [SF_MAP[k] for k, v in snr.items() if np.isfinite(v) and v < 1.0]
        print(f"T = {T} 交易日" + (f"   ⚠ 統計功效不足: {', '.join(low)}" if low else ""))
        print(df.round(3).to_string())
        df.round(4).to_csv(f"{OUT}/tables/scores_{wname.replace(' ', '_')}.csv")
    pd.DataFrame(allsnr).T.rename(columns=SF_MAP).round(3).to_csv(f"{OUT}/tables/signal_to_noise.csv")
    print(f"\n全部表格已存於 {OUT}/tables/")
