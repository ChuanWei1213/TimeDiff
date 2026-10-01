"""Configuration for the Financial Fidelity Evaluation Framework.

所有可調參數集中在此。修改模型清單或評估期間只需要改這個檔案。
設計依據見 ../evaluation_framework_design.md
"""
import os

# ── 專案路徑 ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR     = os.path.join(PROJECT_ROOT, "datasets", "prediction", "data_finance")
REAL_RETURNS_CSV = os.path.join(DATA_DIR, "dow10_wf_returns.csv")

RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
FIG_DIR      = os.path.join(RESULTS_DIR, "figures")
FIGDATA_DIR  = os.path.join(RESULTS_DIR, "figure_data")
TABLE_DIR    = os.path.join(RESULTS_DIR, "tables")

# ── 標的 ────────────────────────────────────────────────────
TICKERS = ["AAPL", "AMGN", "CRM", "CSCO", "IBM", "INTC", "MSFT", "NKE", "VZ", "WMT"]
N_TICKER = len(TICKERS)

# ── 評估期間 ────────────────────────────────────────────────
# PRIMARY：主表排名唯一依據。選擇理由完全基於真實市場資料的統計性質
# （regime 覆蓋、訊噪比、尾部樣本數），在計算任何模型分數之前固定。
PRIMARY_EVAL_START = "2018-01-02"
PRIMARY_EVAL_END   = "2024-12-31"

# ROBUSTNESS：僅用於分析模型在不同 market regime 下的穩健性，
# 不參與主要模型排名。
ROBUSTNESS_WINDOWS = {
    "2020 Crisis": ("2020-01-01", "2020-12-31"),
    "2022 Bear":   ("2022-01-01", "2022-12-31"),
}

EVAL_WINDOWS = {"2018-2024 (Primary)": (PRIMARY_EVAL_START, PRIMARY_EVAL_END)}
EVAL_WINDOWS.update(ROBUSTNESS_WINDOWS)
PRIMARY_WINDOW_KEY = "2018-2024 (Primary)"

# ── Metric 參數 ─────────────────────────────────────────────
# Hill：k = ceil(n_neg ** HILL_K_EXPONENT)。2/3 是尾部指數估計的常用取捨，
# 兼顧偏誤（k 太大 → 納入非尾部觀測）與變異（k 太小 → 估計噪音爆炸）。
HILL_K_EXPONENT = 2.0 / 3.0
HILL_K_SENSITIVITY = (0.5, 1.0, 2.0)   # 報告 k 的敏感度倍率

# Jump：門檻的機率水平。以「曝險數」而非結果決定：
# q95 在 T=1761 下每檔約 88 筆，q99 僅 18 筆。
JUMP_Q = 0.95
JUMP_Q_SENSITIVITY = (0.95, 0.99)

# VC / LE 曲線的 lag 上限。必須 < walk-forward 的 PRED_LEN=21，
# 因為 lag >= 21 的相依性有一部分來自真實 condition 而非模型本身。
ACF_MAX_LAG = 20
WALKFORWARD_PRED_LEN = 21
assert ACF_MAX_LAG < WALKFORWARD_PRED_LEN, "K 必須小於生成區塊長度，否則測到的是協定假象"

# Tail dependence 的機率水平（copula 版與 absolute 版共用）
TAIL_DEP_Q = 0.05
TAIL_DEP_MIN_EXCEEDANCE = 20   # 低於此曝險數 → 標記為不可靠

# ── Bootstrap ───────────────────────────────────────────────
BOOTSTRAP_B = 500          # tau 校準的重抽次數
BOOTSTRAP_CI_B = 500       # 信賴區間的重抽次數
BOOTSTRAP_SEED = 20240816
# stationary block bootstrap 的平均 block 長度 = T ** BLOCK_LEN_EXPONENT
BLOCK_LEN_EXPONENT = 1.0 / 3.0

# 訊噪比低於此值 → 該 metric 在該 window 下統計功效不足，表格標記
MIN_SIGNAL_TO_NOISE = 1.0

# ── Score 轉換 ──────────────────────────────────────────────
# Score = 2 ** (-D / tau)：D=0 -> 1，D=tau -> 0.5。
# tau 只由真實資料的 block bootstrap 得出，不使用任何 synthetic 資料。
SCORE_KERNEL = "halving"          # "halving" | "gaussian"
SCORE_KERNEL_SENSITIVITY = ("halving", "gaussian")

# ── 權重 ────────────────────────────────────────────────────
# Hierarchical：兩個概念軸對等，避免權重被「各軸底下我們碰巧定義了幾個指標」決定。
SF_WEIGHTING = "hierarchical"     # "hierarchical" | "equal"
W_DISTRIBUTION = 0.5
W_DYNAMICS     = 0.5
DISTRIBUTION_METRICS = ["heavy_tail", "skewness"]
DYNAMICS_METRICS     = ["jump", "vol_clustering", "leverage"]

# Cross-asset：等權（無理論依據認定何者更重要）
CROSS_ASSET_METRICS = ["correlation", "pca", "tail_dep_copula"]
# 事先設定的冗餘規則：若 corr 與 pca 的 score 在所有模型上的相關 > 此值，
# 改用 corr+pca 合計 0.5、tail_dep 0.5，避免相關結構被雙重計分。
REDUNDANCY_CORR_THRESHOLD = 0.9

# Overall：univariate 與 multivariate fidelity 是互補且不可互相取代的品質軸
W_SF = 0.5
W_CA = 0.5

# ── 模型清單 ────────────────────────────────────────────────
# None = 自動掃描 outputs/。要限定論文用的模型時填入 tag 清單。
MODELS_WHITELIST = ["dow10_wf_v2","dow10_wf_vpred","v_cross","v_cross_ga","v_whiten","v_3layer","v_4layer","v_4layer_fix","gauss_sigma","garch_null","cndiff","nsdiff","armd"]
MODEL_DISPLAY_NAMES = {
    "dow10_wf_v2": "Baseline (TimeDiff walk-forward)",
    "dow10_wf_vpred": "TimeDiff (v-prediction)",
    "v_cross":    "v + Cross Loss",
    "v_cross_ga": "v + High-Noise Cross Loss",
    "v_whiten":   "v + Cross-Asset Whitening",
    "v_3layer":   "v + Whiten + VolScale",
    "v_4layer":   "v + Whiten + VolScale + HeavyTail",
    "v_4layer_fix": "v + 4-Layer (corr-fixed)",
    "gauss_sigma": "[NULL] Gaussian(Sigma)",
    "garch_null":  "[NULL] GARCH+Sigma+t",
    "dow10_wf":    "Baseline v1 (single path)",
    "cndiff":"CN-Diff",
    "nsdiff":"NS-Diff",
    "armd":"ARMD"
}

# 只有使用 walk-forward rolling protocol、且覆蓋 PRIMARY window 的模型
# 才能進入主表（Option 1：所有 Method 必須在完全相同的時間區間下比較）。
REQUIRE_WALKFORWARD_PROTOCOL = True
