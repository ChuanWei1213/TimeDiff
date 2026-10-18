"""
walk_forward_dow10.py
======================
DOW10 Walk-Forward（滾動重訓）生成實驗：2018-01-02 ~ 2024-12-31。

本檔案完全獨立，**不修改任何既有程式碼**。與既有模組的關係：
  - 重用 exp.exp_main.Exp_Main（僅用其 __init__ / _get_data，訓練迴圈自己寫）
  - 重用 models_diffusion.DDPM 的 pretrain_forward / train_forward / forward
  - 以 monkey-patch 方式把 data_factory.data_dict['custom'] 換成本檔的 DatasetWF
    （既有的 Dataset_Custom 是 70/10/20 比例切分，湊不出 1008/252 的規格）
  - 重用 generate_dow10 的 Stylized Facts 計算與報表

實驗規格
--------
[排程]
  - 預測期：2018-01-02 ~ 2024-12-31，共 1,761 個共同交易日（依實際資料推導）
  - 每 63 個交易日重訓一次，共 28 期；最後一期 60 日
  - 第 k 期 origin = 預測期第 (63k) 天

[每期訓練]
  - 訓練資料嚴格取 [origin-1260, origin)，不含 origin 與未來
  - 選模階段：前 1,008 日訓練 / 後 252 日驗證，scaler 只 fit 前 1,008 日
               依 validation MSE 選最佳 step
  - 正式階段：重新初始化模型與 scaler，用完整 1,260 日（1,177 個訓練窗口）
               訓練到選定的 step
  - 不 warm-start（每期、每階段都重新 Exp_Main(args)），不把生成資料放回訓練窗

[每期生成]
  - 最多 63 日 = 21/21/21 三個區塊；最後一期取前 60 日（21/21/18）
  - 每個 21 日區塊的 condition 都是「當時可取得的真實 63 日」
    block j：cond = real[origin+21j-63 : origin+21j]

[價格重建]
  - 先按日期串接全部 1,761 日生成 returns
  - 唯一初始價 = 2017 最後一個共同交易日的真實 Close
  - Close_t = Close_{t-1} * exp(rclose_t)（全段累積，區塊/重訓邊界都不重置）
  - High_t  = Close_t * exp(rhigh_t)，Low_t = Close_t * exp(rlow_t)（同日 range 特徵）

重要對齊細節
------------
  「step」定義為 optimizer step（global step）。選模階段每 epoch 的 step 數
  (925 窗口 → 28) 與正式階段 (1,177 窗口 → 36) 不同，因此兩階段的
  LR schedule（CosineAnnealingLR）與 Stylized Loss annealing 都以
  T_max = total_steps = --max_steps 為基準、以 step 為單位推進，
  確保「同一個 step」在兩階段有相同的 LR 與 λ，選出的 step 才有意義。

用法
----
  python data_provider/preprocess_walkforward_dow10.py        # 先產生資料
  python walk_forward_dow10.py --tag dow10_wf --max_steps 2000 --eval_every 100
"""

import os
import sys
import gc
import json
import time
import pickle
import random
import argparse
import warnings
from output_layout import walkforward_path, resolve_output_path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath("."))

from utils.timefeatures import time_features
from data_provider import data_factory
from data_provider.data_loader import Dataset_Custom
from exp.exp_main import Exp_Main
from generate_dow10 import compute_stylized_facts, print_and_save_stylized_facts

warnings.filterwarnings("ignore")

# ===========================================================
# 全域常數
# ===========================================================
TICKERS   = ["AAPL", "AMGN", "CRM", "CSCO", "IBM",
             "INTC", "MSFT", "NKE", "VZ", "WMT"]
N_TICKER  = len(TICKERS)
N_FEAT    = 3
NUM_VARS  = N_TICKER * N_FEAT       # 30

SEQ_LEN   = 63
LABEL_LEN = 21
PRED_LEN  = 21

TRAIN_LEN     = 1260                # 每期訓練窗長度
SEL_TRAIN_LEN = 1008                # 選模階段訓練日數
SEL_VAL_LEN   = 252                 # 選模階段驗證日數
STRIDE        = 63                  # 重訓間隔 = 每期生成天數
N_BLOCKS      = STRIDE // PRED_LEN  # 3

PRED_START = "2018-01-02"
PRED_END   = "2024-12-31"

DATA_DIR     = "./datasets/prediction/data_finance"
RETURNS_CSV  = os.path.join(DATA_DIR, "dow10_wf_returns.csv")
OHLC_CSV     = os.path.join(DATA_DIR, "dow10_wf_ohlc.csv")

RCLOSE_IDX = [i * 3 + 0 for i in range(N_TICKER)]
RHIGH_IDX  = [i * 3 + 1 for i in range(N_TICKER)]
RLOW_IDX   = [i * 3 + 2 for i in range(N_TICKER)]

COLS_ORDER = []
for _t in TICKERS:
    COLS_ORDER += [f"{_t}_rclose", f"{_t}_rhigh", f"{_t}_rlow"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ===========================================================
# 1. Walk-Forward 專用 Dataset（monkey-patch 進 data_factory）
# ===========================================================
class DatasetWF(Dataset_Custom):
    """把 Dataset_Custom 的 70/10/20 比例切分換成明確的 train/val 日數。

    args.wf_train_len : 訓練日數（scaler 只 fit 這一段）
    args.wf_val_len   : 驗證日數（0 = 正式階段，不做驗證）

    註：__read_data__ 有兩個結尾底線，不受 Python name mangling 影響
    （mangling 只作用於「兩個前導底線 + 至多一個結尾底線」的名稱），
    因此直接以同名方法覆寫即可蓋掉父類實作。
    """

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = [c for c in df_raw.columns if c not in ("date", self.target)]
        df_raw = df_raw[["date"] + cols + [self.target]]

        n_tr = int(self.args.wf_train_len)
        n_va = int(self.args.wf_val_len)
        assert len(df_raw) == n_tr + n_va, (
            f"視窗 CSV 長度 {len(df_raw)} != wf_train_len+wf_val_len "
            f"({n_tr}+{n_va})；請確認視窗切片正確。"
        )

        # train = [0, n_tr)；val 的輸入可回看訓練尾端 seq_len 天，
        # 但其預測目標全部落在驗證期內（與既有 loader 同一慣例，無洩漏）。
        border1s = [0,    n_tr - self.seq_len, n_tr - self.seq_len]
        border2s = [n_tr, n_tr + n_va,         n_tr + n_va]
        if n_va == 0:
            # 正式階段沒有驗證集；退化成訓練尾端一個窗口，避免長度為負
            border1s[1] = border1s[2] = n_tr - self.seq_len - self.pred_len
            border2s[1] = border2s[2] = n_tr
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ("M", "MS"):
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values[:n_tr])     # ← 只 fit 訓練段
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"]   = df_stamp.date.apply(lambda r: r.month)
            df_stamp["day"]     = df_stamp.date.apply(lambda r: r.day)
            df_stamp["weekday"] = df_stamp.date.apply(lambda r: r.weekday())
            df_stamp["hour"]    = df_stamp.date.apply(lambda r: r.hour)
            data_stamp = df_stamp.drop(columns=["date"]).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y))


data_factory.data_dict["custom"] = DatasetWF   # ← 不動原檔的替換


# ===========================================================
# 2. args（訓練 + 推論共用；欄位與 main_ddpm.py / generate_dow10.py 對齊）
# ===========================================================
def build_args(cli):
    class Args:
        pass
    args = Args()

    # ── 基本 ──
    args.model        = "DDPM"
    args.tag          = cli.tag
    args.is_training  = 1
    args.stage_mode   = "TWO"
    args.out_figures  = 0
    args.vis_ar_part  = 0
    args.vis_MTS_analysis = 0
    args.sample_times = 1
    args.beta_dist_alpha = -1
    args.our_ddpm_clip   = cli.ddpm_clip

    # ── 視窗 ──
    args.seq_len   = SEQ_LEN
    args.label_len = LABEL_LEN
    args.pred_len  = PRED_LEN

    # ── 資料 ──
    args.dataset_name = "custom"
    args.data         = "custom"
    args.weather_type = "mintemp"
    args.root_path    = None          # 每期指向該期的視窗 CSV
    args.data_path    = None
    args.features     = "M"
    args.target       = "WMT_rlow"    # 必須是 CSV 最後一欄（見 main() 的 assert）
    args.num_vars     = NUM_VARS
    args.checkpoints  = "./checkpoints/"
    args.freq         = "b"
    args.use_valset   = True
    args.embed        = "timeF"
    args.num_workers  = 0
    args.itr          = 1
    args.extreme_upsample_ratio     = 1
    args.extreme_upsample_threshold = 0.95
    # walk-forward 切分參數（由 DatasetWF 讀取，每階段改寫）
    args.wf_train_len = SEL_TRAIN_LEN
    args.wf_val_len   = SEL_VAL_LEN

    # ── Diffusion ──
    args.interval   = 1000
    args.ot_ode     = True
    args.beta_max   = 0.3
    args.t0         = 1e-4
    args.T          = 1.0
    args.nfe        = 100
    args.dim_LSTM   = 64
    args.diff_steps = cli.diff_steps
    args.UNet_Type  = "CNN"
    args.D3PM_kernel_size = 5
    args.use_freq_enhance = 0
    args.type_sampler     = cli.type_sampler
    args.parameterization = cli.parameterization
    args.use_xt_residual  = cli.use_xt_residual
    args.noise_dist       = cli.noise_dist
    args.noise_df         = cli.noise_df
    args.use_window_normalization = cli.use_window_normalization

    args.model_channels          = 256
    args.ddpm_inp_embed          = 256
    args.ddpm_dim_diff_steps     = 256
    args.ddpm_channels_conv      = 256
    args.ddpm_channels_fusion_I  = 256
    args.ddpm_layers_inp         = 5
    args.ddpm_layers_I           = 5
    args.ddpm_layers_II          = 5
    args.cond_ddpm_num_layers    = 5
    args.cond_ddpm_channels_conv = 64

    args.ablation_study_case         = "none"
    args.weight_pred_loss            = 0.0
    args.ablation_study_F_type       = "CNN"
    args.ablation_study_masking_type = "none"
    args.ablation_study_masking_tau  = 0.9

    # ── Stylized Facts Loss（與 run_dow10.sh 預設一致）──
    args.use_volatility_loss  = cli.use_volatility_loss
    args.use_leverage_loss    = cli.use_leverage_loss
    args.use_fat_tail_loss    = False
    args.use_gpd_tail_loss    = False
    args.anneal_stylized_loss = True
    args.anneal_linear        = True
    args.lambda_vol_max       = cli.lambda_vol_max
    args.lambda_lev_max       = cli.lambda_lev_max
    args.lambda_tail_max      = 0.05
    args.lambda_gpd_max       = 0.0
    args.gpd_threshold_q      = 0.90
    args.gpd_soft_tau         = 0.10
    args.anneal_ratio         = 0.2
    args.tail_anneal_delay    = 2.0
    args.tail_anneal_ratio    = 0.4
    args.stylized_max_lag     = 10
    args.leverage_window      = 5
    args.stylized_timestep_ratio = 0.5

    # ── Cross-Asset Dependency Loss（預設關閉；λ=0 即完全等同原模型）──
    args.aux_no_teacher_forcing = cli.aux_no_teacher_forcing
    args.lambda_cross_max     = cli.lambda_cross_max
    args.cross_min_t_ratio    = cli.cross_min_t_ratio
    args.cross_gen_aligned    = cli.cross_gen_aligned
    args.cross_gen_t_ratio    = cli.cross_gen_t_ratio
    args.cross_timestep_ratio = cli.cross_timestep_ratio

    # ── Energy Score（對付 variance collapse；預設關閉）──
    args.lambda_energy_max  = cli.lambda_energy_max
    args.energy_n_samples   = cli.energy_n_samples
    args.energy_min_t_ratio = cli.energy_min_t_ratio
    args.energy_beta        = cli.energy_beta

    # ── 訓練 ──
    args.learning_rate   = cli.learning_rate
    args.batch_size      = cli.batch_size
    args.test_batch_size = cli.test_batch_size
    args.use_amp         = cli.use_amp and torch.cuda.is_available()
    args.train_epochs    = 1        # 佔位：本檔以 step 為單位，不使用
    args.pretrain_epochs = 1        # 佔位
    args.early_stop_patience = 10
    args.save_last_epoch = False

    # ── 裝置 ──
    args.use_gpu       = torch.cuda.is_available()
    args.gpu           = 0
    args.use_multi_gpu = False
    args.devices       = "0"
    args.seed          = cli.seed
    args.device        = DEVICE

    args.model_id = f"custom_{SEQ_LEN}_{PRED_LEN}"
    return args



# ===========================================================
# 2.5 Cross-Asset 白化（correlated noise prior 的等價實作）
# ===========================================================
# 動機：擴散注入的噪聲 eps ~ N(0, I) 在通道之間獨立，而診斷顯示四個模型的
#       「噪聲成分」跨資產相關全部 ~0（0.009~0.013）—— denoiser 從來沒有學會
#       把獨立噪聲映射成相關輸出。跨資產相關只能來自條件均值這一個來源。
#
# 作法：令訓練窗的 rclose 共變異矩陣 Sigma = L L^T。
#       訓練前白化 z = r @ inv(L)^T（cov(z) = I，相關被剝離）；
#       生成後著色 r = z @ L^T（等價於噪聲先驗 N(0, Sigma)）。
#
# 為什麼用資料變換而不是直接改 _draw_noise：
#       DDPM 的 posterior 係數與 DPM-Solver 的 probability-flow ODE 都是在
#       「各向同性噪聲」下推導的；直接把 eps 換成相關噪聲會讓那些公式失效。
#       在 z 空間裡噪聲仍是各向同性，因此擴散數學、v-parameterization、
#       DPM-Solver 完全不需要改動，而 L·N(0,I) = N(0, Sigma) 在數學上等價。
#
# 只作用於 10 個 rclose 通道：rhigh/rlow 有符號約束（rhigh>=0>=rlow），
# 白化會破壞它，著色也不保證還原，因此維持原樣。
#
# 驗證過的性質（見對話紀錄）：往返可逆到 4e-17；白化不會讓尾巴變薄
# （Hill alpha 2.610 -> 2.470），因為 Cholesky 是逐次剝除已解釋成分、
# 保留個股特異衝擊，而不是把 10 檔平均。

def fit_whitener(arr30):
    """由訓練窗估計 rclose 的 Cholesky 因子 L。arr30: (T, 30) 原始尺度報酬。"""
    r = np.asarray(arr30, dtype=np.float64)[:, RCLOSE_IDX]
    cov = np.cov(r, rowvar=False)
    # 輕微 ridge：避免 1260x10 估計在極端市況下接近奇異
    cov += np.eye(N_TICKER) * 1e-12 * np.trace(cov) / N_TICKER
    return np.linalg.cholesky(cov)


def fit_whitener_corr(arr30):
    """由**相關矩陣**（非共變異）估 Cholesky 因子。

    用於著色端：白化空間的 z 已是單位變異數，注入相關時只需要相關結構，
    不能連原始報酬的尺度一起帶進來。
    """
    r = np.asarray(arr30, dtype=np.float64)[:, RCLOSE_IDX]
    C_ = np.corrcoef(r, rowvar=False)
    C_ += np.eye(N_TICKER) * 1e-10
    return np.linalg.cholesky(C_)


def whiten_rclose(arr30, L):
    """z = r @ inv(L)^T，只換 rclose 通道；其餘原樣。"""
    out = np.array(arr30, dtype=np.float64, copy=True)
    out[..., RCLOSE_IDX] = out[..., RCLOSE_IDX] @ np.linalg.inv(L).T
    return out


def color_rclose(arr30, L):
    """r = z @ L^T，whiten_rclose 的逆運算。"""
    out = np.array(arr30, dtype=np.float64, copy=True)
    out[..., RCLOSE_IDX] = out[..., RCLOSE_IDX] @ L.T
    return out



# ===========================================================
# 2.6 時變波動縮放（波動聚集 / 厚尾的結構性保證）
# ===========================================================
# 動機：診斷顯示 v-prediction 的生成序列 Ljung-Box Q(20)=43，與 i.i.d. 高斯（39）
#       無異，而真實資料是 455 —— 模型完全沒有學到波動聚集。原因是擴散噪聲的
#       尺度固定，要產生聚集必須讓輸出振幅隨 condition 變化，而 MSE 目標只獎勵
#       條件均值、不獎勵條件變異數。
#
# 作法：r_t = sigma_t * z_t。sigma_t 由 GARCH(1,1) 遞迴，模型只生成標準化的 z。
#       訓練前 z = r / sigma(真實過去)；生成後 r = sigma(已生成過去) * z。
#
# 為什麼用 GARCH 而非 EWMA：EWMA 的 sigma^2 是鞅（隨機遊走），生成時遞迴會發散
#       —— 實測 lambda=0.94 讓 LB Q 衝到 2611（真實 455）。GARCH(1,1) 在
#       alpha+beta<1 時有穩態 omega/(1-alpha-beta)，會均值回歸。
#
# 為什麼這比白化更站得住腳：白化的 Sigma_hat 由真實資料估出並注入；而生成時的
#       sigma_t 完全由**模型自己的輸出**遞迴產生，只有 (omega,alpha,beta) 來自
#       訓練窗。生成過程不看任何真實資料，因此沒有「看答案」的疑慮。
#
# 驗證過的性質（見對話紀錄）：真實報酬經 sigma_t 標準化後 LB Q 由 455 降到 5
#       （時序結構被完整吸收）；而 sigma_t(遞迴) x 高斯 z 可還原 Hill alpha 2.33、
#       超額峰度 9.88（真實 2.61 / 10.70）—— 厚尾是波動混合的結構性後果，
#       模型不需要學重尾噪聲。

def _garch_nll(par, r):
    om, al, be = np.exp(par)                       # 保證正值
    if al + be >= 0.999:
        return 1e10
    v = r.var()
    nll = 0.0
    for x in r:
        nll += np.log(v + 1e-18) + x * x / (v + 1e-18)
        v = om + al * x * x + be * v
    return 0.5 * nll


def _garch_nll_pooled(par, R):
    """共用 (alpha, beta)、逐檔 omega 的 QMLE 負對數概似。R: (T, 10)。"""
    al, be = 1.0 / (1.0 + np.exp(-par[:2]))        # sigmoid -> (0,1)
    if al + be >= 0.995:
        return 1e10
    om = np.exp(par[2:])                            # 逐檔
    v = R.var(axis=0)
    nll = 0.0
    for t in range(R.shape[0]):
        x2 = R[t] ** 2
        nll += np.sum(np.log(v + 1e-18) + x2 / (v + 1e-18))
        v = om + al * x2 + be * v
    return 0.5 * nll


def fit_garch(arr30, pooled=True):
    """在**訓練窗**上以 QMLE 擬合 GARCH(1,1)。回傳 (10, 3) 的 (omega, alpha, beta)。

    pooled=True：alpha / beta 跨 10 檔共用，omega 逐檔。

    為什麼要 pooled：逐檔估計在 1,260 天上的持續性參數雜訊極大 —— 實測 alpha+beta
    的逐檔範圍是 0.40~0.997，平均被拉低到 0.85，使遞迴出的 sigma_t 均值回歸過快、
    做不出波動聚集。共用持續性把觀測數從 1,260 提升到 12,600，是金融 GARCH 的
    標準做法（個股持續性估計已知有向下偏誤）。
    參數回收測試（模擬已知 GARCH）顯示擬合器本身無誤：alpha+beta 誤差 0.005~0.013。
    """
    from scipy.optimize import minimize
    R = np.asarray(arr30, dtype=np.float64)[:, RCLOSE_IDX]
    v0 = R.var(axis=0)
    if pooled:
        x0 = np.concatenate([[np.log(0.06 / (1 - 0.06)), np.log(0.92 / (1 - 0.92))],
                             np.log(v0 * 0.02)])
        try:
            res = minimize(_garch_nll_pooled, x0, args=(R,), method="Nelder-Mead",
                           options=dict(maxiter=4000, maxfev=8000, xatol=1e-7, fatol=1e-7))
            al, be = 1.0 / (1.0 + np.exp(-res.x[:2]))
            om = np.exp(res.x[2:])
        except Exception:
            al, be, om = 0.06, 0.92, v0 * 0.02
        if not np.isfinite([al, be]).all() or al + be >= 0.995 or al + be < 0.5:
            al, be, om = 0.06, 0.92, v0 * 0.02
        return np.column_stack([om, np.full(N_TICKER, al), np.full(N_TICKER, be)])

    out = np.zeros((N_TICKER, 3))
    for j in range(N_TICKER):
        r = R[:, j]
        x0 = np.log([v0[j] * 0.05, 0.08, 0.90])
        try:
            res = minimize(_garch_nll, x0, args=(r,), method="Nelder-Mead",
                           options=dict(maxiter=600, xatol=1e-6, fatol=1e-6))
            om, al, be = np.exp(res.x)
        except Exception:
            om, al, be = v0[j] * 0.05, 0.08, 0.90
        if al + be >= 0.999 or not np.isfinite([om, al, be]).all():
            om, al, be = v0[j] * 0.05, 0.08, 0.90
        out[j] = (om, al, be)
    return out


def garch_sigma_causal(arr30, par, v_init=None):
    """由**已觀測**的報酬算 sigma_t（嚴格因果：sigma_t 只用 t-1 以前）。

    回傳 (T, 10) 的 sigma 與最後的 variance 狀態（供跨期接續）。
    """
    r = np.asarray(arr30, dtype=np.float64)[:, RCLOSE_IDX]
    T = r.shape[0]
    sig = np.empty((T, N_TICKER))
    om, al, be = par[:, 0], par[:, 1], par[:, 2]
    v = om / np.maximum(1 - al - be, 1e-6) if v_init is None else np.array(v_init, dtype=np.float64)
    for t in range(T):
        sig[t] = np.sqrt(np.maximum(v, 1e-16))
        v = om + al * r[t] ** 2 + be * v
    return sig, v


def fit_vol_structure(arr30, par):
    """在訓練窗上估計兩個結構參數，兩者都由資料決定、不調參。

    rho : 波動的「共同成分」比例 = 各檔 |r| 的平均兩兩相關。
          直接量的就是「波動是否同步漲跌」。sigma_t 若逐檔獨立遞迴，
          10 條 sigma 路徑各走各的，會像乘法噪聲一樣稀釋掉已建好的跨資產相關
          （實測 0.408 -> 0.295）。加入共同驅動可保住相關結構。
    df  : 標準化殘差 z = r/sigma 的 Student-t 自由度，由超額峰度反推
          df = 6/exKurt + 4。真實資料 z 的 exKurt 約 14.7 -> df 約 4.4。
    """
    from scipy import stats as _st
    R = np.asarray(arr30, dtype=np.float64)[:, RCLOSE_IDX]
    a = np.abs(R)
    C = np.corrcoef(a, rowvar=False)
    iu = np.triu_indices(N_TICKER, 1)
    rho = float(np.clip(np.nanmean(C[iu]), 0.0, 0.95))

    sig, _ = garch_sigma_causal(arr30, par)
    z = R / np.maximum(sig, 1e-12)
    ek = float(np.nanmean([_st.kurtosis(z[:, j]) for j in range(N_TICKER)]))
    df = float(np.clip(6.0 / max(ek, 1e-6) + 4.0, 3.0, 30.0))

    m = R.mean(axis=1)
    return dict(rho=rho, df=df, v_stat=par[:, 0] / np.maximum(1 - par[:, 1] - par[:, 2], 1e-6),
                v_m=float(m.var()))


def calibrate_marginal_t(z, df):
    """把 z 的邊際校準成 Student-t(df)，**保序**（單調變換）。

    模型產生的 z 邊際接近高斯（實測 exKurt 0.2），但真實標準化殘差 exKurt 約 14.7。
    用 rank -> uniform -> t 分位數的映射把邊際換掉，而單調變換**完全保留 copula**
    （跨資產相依與時序順序都不變），因此不會破壞前面幾層建立的結構。

    這與直接改擴散的噪聲先驗不同：DPM-Solver 是確定性 ODE，輸出是 x_T 的確定性
    函數，把 x_T 換成重尾**不保證**輸出邊際也是重尾。分位數映射則是直接、可驗證的。
    """
    from scipy import stats as _st
    out = np.array(z, dtype=np.float64, copy=True)
    x = out[:, RCLOSE_IDX]
    T = x.shape[0]
    scale = np.sqrt(max(df - 2.0, 1e-6) / df)          # 正規化到單位變異數
    for j in range(x.shape[1]):
        r = np.argsort(np.argsort(x[:, j])) + 1.0
        u = np.clip(r / (T + 1.0), 1e-6, 1 - 1e-6)
        x[:, j] = _st.t.ppf(u, df) * scale
    out[:, RCLOSE_IDX] = x
    return out


def descale_recursive(z30, par, v_init, clamp=(0.15, 8.0), struct=None):
    """z -> r：sigma_t 由**已生成**的報酬遞迴，因此波動聚集是自我維持的。

    clamp 以穩態波動的倍數限制 sigma，避免回饋迴路失控（GARCH 已有均值回歸，
    這是額外的安全網）。回傳 (T, 30)，非 rclose 通道原樣。
    """
    out = np.array(z30, dtype=np.float64, copy=True)
    z = out[:, RCLOSE_IDX]
    T = z.shape[0]
    om, al, be = par[:, 0], par[:, 1], par[:, 2]
    v_ss = om / np.maximum(1 - al - be, 1e-6)          # 穩態變異數
    lo, hi = clamp[0] ** 2 * v_ss, clamp[1] ** 2 * v_ss
    rho = 0.0 if struct is None else float(struct["rho"])
    v_m = 1.0 if struct is None else max(float(struct["v_m"]), 1e-18)
    v = np.array(v_init, dtype=np.float64)
    r = np.empty_like(z)
    for t in range(T):
        v = np.clip(v, lo, hi)
        r[t] = np.sqrt(v) * z[t]
        # 波動的驅動項拆成「個股特異」與「市場共同」兩部分。
        # rho=0 時退化成標準 GARCH（逐檔獨立），rho>0 讓 10 檔的 sigma 同步起伏，
        # 這才是真實市場的結構，也避免獨立的 sigma 路徑稀釋跨資產相關。
        shock = r[t] ** 2
        if rho > 0:
            m2 = r[t].mean() ** 2
            shock = (1 - rho) * shock + rho * (v_ss / v_m) * m2
        v = om + al * shock + be * v
    out[:, RCLOSE_IDX] = r
    return out, v


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ===========================================================
# 3. 訓練（自寫迴圈，以 step 為單位；不呼叫 Exp_Main.train）
# ===========================================================
def _to_device(batch, device):
    bx, by, bxm, bym = batch[0], batch[1], batch[2], batch[3]
    return (bx.float().to(device), by.float().to(device),
            bxm.float().to(device), bym.float().to(device))


def pretrain_dlinear(exp, train_loader, n_steps, lr=1e-4, log_every=200):
    """Stage TWO 的 DLinear 預訓（等價於 Exp_Main.pretrain，但以 step 計）。

    權重留在記憶體裡直接進下一階段，因此不需要存 / 讀 pretrain checkpoint。
    """
    if n_steps <= 0:
        return
    assert len(train_loader) > 0, "train_loader 為空，無法預訓。"
    optim_ = torch.optim.Adam(exp.model.parameters(), lr=lr)
    step, losses, t0 = 0, [], time.time()
    while step < n_steps:
        for batch in train_loader:
            bx, by, bxm, bym = _to_device(batch, exp.device)
            optim_.zero_grad()
            loss = exp.model.pretrain_forward(bx, bxm, by, bym)
            loss.backward()
            optim_.step()
            losses.append(loss.item())
            step += 1
            if step % log_every == 0:
                print(f"    [pretrain] step {step}/{n_steps} "
                      f"loss={np.mean(losses[-log_every:]):.6f}")
            if step >= n_steps:
                break
    print(f"    [pretrain] done {n_steps} steps, "
          f"loss={np.mean(losses[-50:]):.6f}, {time.time()-t0:.1f}s")


@torch.no_grad()
def val_mse(exp, val_loader, max_batches=0):
    """完整（或前 max_batches 個 batch 的）驗證 MSE。

    刻意不用 Exp_Main.val()：它有 `if i > 5: break` 且沒有包 no_grad，
    不適合當選模依據。
    """
    exp.model.eval()
    sse, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        bx, by, bxm, bym = _to_device(batch, exp.device)
        outputs, _, true, _, _ = exp.model.forward(bx, bxm, by, bym, sample_times=1)
        if outputs.dim() == 4:
            outputs = outputs.mean(dim=1)
        sse += ((outputs - true) ** 2).sum().item()
        n   += true.numel()
        if max_batches and (i + 1) >= max_batches:
            break
    exp.model.train()
    return sse / max(n, 1)


def train_diffusion(exp, train_loader, max_steps, target_steps=None,
                    val_loader=None, eval_every=0, val_batches=0, log_every=100):
    """訓練擴散模型至 target_steps（預設 = max_steps）。

    max_steps    : LR schedule 與 stylized annealing 的基準（兩階段必須相同）
    target_steps : 實際停止的 step（正式階段 = 選模選出的 best_step）
    回傳 (best_step, val_history)；沒有 val_loader 時 best_step = target_steps。
    """
    target_steps = int(target_steps or max_steps)
    assert len(train_loader) > 0, "train_loader 為空，無法訓練。"

    optim_ = torch.optim.Adam(exp.model.parameters(), lr=exp.args.learning_rate)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=max(1, max_steps))
    amp_scaler = torch.cuda.amp.GradScaler() if exp.args.use_amp else None

    if hasattr(exp.model, "diffusion_worker"):
        exp.model.diffusion_worker.total_steps = max(1, max_steps)
        exp.model.diffusion_worker.global_step = 0

    best = (float("inf"), 0)     # (val_mse, step)
    history, losses = [], []
    step, t0 = 0, time.time()

    exp.model.train()
    while step < target_steps:
        for batch in train_loader:
            bx, by, bxm, bym = _to_device(batch, exp.device)
            optim_.zero_grad()

            if exp.args.use_amp:
                with torch.cuda.amp.autocast():
                    loss = exp.model.train_forward(bx, bxm, by, bym)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optim_)
                amp_scaler.update()
            else:
                loss = exp.model.train_forward(bx, bxm, by, bym)
                loss.backward()
                optim_.step()

            sched.step()
            losses.append(loss.item())
            step += 1

            if step % log_every == 0:
                print(f"    [train] step {step}/{target_steps} "
                      f"loss={np.mean(losses[-log_every:]):.6f} "
                      f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)")

            if val_loader is not None and eval_every and step % eval_every == 0:
                mse = val_mse(exp, val_loader, max_batches=val_batches)
                history.append({"step": step, "val_mse": mse})
                mark = ""
                if mse < best[0]:
                    best = (mse, step)
                    mark = "  ← best"
                print(f"    [valid] step {step}  val_mse={mse:.7f}{mark}")

            if step >= target_steps:
                break

    if val_loader is None or not history:
        best = (float("nan"), target_steps)
    print(f"    [train] finished at step {step}, best_step={best[1]}, "
          f"best_val_mse={best[0]:.7f}, {time.time()-t0:.0f}s")
    return best[1], history


# ===========================================================
# 4. 生成單一 21 日區塊
# ===========================================================
@torch.no_grad()
def generate_block(exp, cond_scaled, n_paths=1, vol_scale=1.0):
    """cond_scaled: (SEQ_LEN, NUM_VARS) 已標準化的真實 condition
       回傳 (n_paths, PRED_LEN, NUM_VARS)，仍在標準化空間。

    每個區塊的 condition 都是真實資料（不是前一段的生成結果），因此
    n_paths 條路徑在給定同一 condition 下是條件獨立的 —— 直接把 condition
    複製到 batch 維、一次取樣即可，比 sample_times 的序列迴圈快得多。
    """
    exp.model.eval()
    cond_b = np.repeat(cond_scaled[np.newaxis], n_paths, axis=0)
    x_enc = torch.tensor(cond_b, dtype=torch.float32).to(exp.device)

    dec_inp = np.concatenate(
        [cond_scaled[-LABEL_LEN:], np.zeros((PRED_LEN, NUM_VARS), dtype=np.float32)], axis=0)
    x_dec = torch.tensor(np.repeat(dec_inp[np.newaxis], n_paths, axis=0),
                         dtype=torch.float32).to(exp.device)

    x_mark_enc = torch.zeros(n_paths, SEQ_LEN, 4, dtype=torch.float32).to(exp.device)
    x_mark_dec = torch.zeros(n_paths, LABEL_LEN + PRED_LEN, 4, dtype=torch.float32).to(exp.device)

    outputs, _, _, _, _ = exp.model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, sample_times=1)
    out = outputs[:, 0] if outputs.dim() == 4 else outputs      # (n_paths, PRED_LEN, NUM_VARS)
    pred = out.cpu().numpy()

    if vol_scale != 1.0:   # 振幅校準（與 generate_dow10.py 同一作法，逐路徑逐區塊）
        m = pred.mean(axis=1, keepdims=True)
        pred = m + (pred - m) * vol_scale
    return pred


# ===========================================================
# 5. 價格重建 / 繪圖
# ===========================================================
def reconstruct_ohlc(all_ret, seed_close):
    """all_ret: (T, NUM_VARS) 生成的 log-return；seed_close: (N_TICKER,)
       全段一次累積，區塊與重訓邊界都不用真實價格重置。

    rhigh = log(H/C) 依定義恆 >= 0、rlow = log(L/C) 恆 <= 0，但模型是把 30 個
    欄位當獨立無界通道生成，沒有任何符號約束；真實資料裡 0 只離 rhigh 均值約
    1 個標準差，所以偶爾（實測 ~0.003% 的格子，且集中在採樣爆掉的區塊）會跨到
    另一側，造成 low <= close <= high 不成立。這裡把違反的值投影回合法域
    （取最近的合法點），保證輸出的 OHLC 一定滿足 low <= close <= high。"""
    T = all_ret.shape[0]
    close = np.zeros((T, N_TICKER))
    prev = np.asarray(seed_close, dtype=np.float64)
    for t in range(T):
        prev = prev * np.exp(all_ret[t, RCLOSE_IDX])
        close[t] = prev
    high = close * np.exp(np.maximum(all_ret[:, RHIGH_IDX], 0.0))
    low  = close * np.exp(np.minimum(all_ret[:, RLOW_IDX],  0.0))
    return close, high, low


def to_long_format(dates, close, high, low):
    """轉成下游常用的長格式：date,tic,close,high,low。

    每個交易日 N_TICKER 列，先依日期、再依 ticker 排序（FinRL 等下游的慣例）。
    close/high/low: (T, N_TICKER)
    """
    n_t = len(dates)
    return pd.DataFrame({
        "date":  np.repeat([d.strftime("%Y-%m-%d") for d in dates], N_TICKER),
        "tic":   np.tile(TICKERS, n_t),
        "close": close.reshape(-1),
        "high":  high.reshape(-1),
        "low":   low.reshape(-1),
    })


def plot_prices(dates, syn_close, real_close, origins_dates, out_path, extra_closes=()):
    fig, axes = plt.subplots(2, 5, figsize=(24, 9))
    axes = axes.flatten()
    for i, (ax, ticker) in enumerate(zip(axes, TICKERS)):
        for c in extra_closes:                      # 其他路徑，淡色顯示離散度
            ax.plot(dates, c[:, i], lw=0.5, color="tab:orange", alpha=0.25)
        ax.plot(dates, real_close[:, i], lw=1.0, color="tab:blue", alpha=0.85, label="Real")
        ax.plot(dates, syn_close[:, i],  lw=1.0, color="tab:orange", label="Synthetic")
        for d in origins_dates:
            ax.axvline(d, color="gray", lw=0.4, alpha=0.25)
        ax.set_title(ticker, fontsize=10)
        ax.set_ylabel("Price (USD)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    # 標題用英文，避免多數環境缺 CJK 字型而出現豆腐方塊
    fig.suptitle("DOW10 Walk-Forward: Synthetic vs Real Close (2018-2024)\n"
                 "grey lines = 63-day retrain boundaries (prices are NOT reset)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"✅ 股價走勢圖已儲存：{out_path}")


# ===========================================================
# 6. 單期（一個 origin）流程
# ===========================================================
def generate_cycle(exp, scaler, cli, k, origin_idx, cycle_len, feat_df,
                   whitener=None, garch=None, colorizer=None):
    """用已訓練好的該期模型生成 21/21/21，回傳 (n_paths, cycle_len, NUM_VARS)（原始尺度）。

    whitener 不是 None 時，condition 先白化（與訓練資料同一個變換），
    生成結果在 inverse_transform 之後再著色回報酬空間。
    """
    set_seed(cli.seed + 7777 * (cli.gen_seed + 1) + k)   # 生成亂數與訓練分離，可重現
    blocks = []
    for j in range(N_BLOCKS):
        s = origin_idx + PRED_LEN * j              # 該區塊第一個預測日
        cond_raw = feat_df.values[s - SEQ_LEN: s]  # 當時可取得的真實 63 日
        if garch is not None:
            # condition 也要進 z 空間；sigma 由「更早的真實資料」因果推進，
            # 與訓練時的標準化方式完全一致（無 look-ahead）。
            warm = feat_df.values[max(0, s - SEQ_LEN - 252): s]
            sg, _ = garch_sigma_causal(warm, garch)
            cond_raw = cond_raw.copy()
            cond_raw[:, RCLOSE_IDX] = cond_raw[:, RCLOSE_IDX] / np.maximum(sg[-SEQ_LEN:], 1e-12)
        if whitener is not None:
            cond_raw = whiten_rclose(cond_raw, whitener)
        cond_scaled = scaler.transform(cond_raw).astype(np.float32)
        pred_scaled = generate_block(exp, cond_scaled,
                                     n_paths=cli.n_paths, vol_scale=cli.vol_scale)
        blocks.append(pred_scaled)
        print(f"    [gen] block {j+1}/{N_BLOCKS} × {cli.n_paths} paths  "
              f"cond {feat_df.index[s-SEQ_LEN].date()}~{feat_df.index[s-1].date()}"
              f" → pred from {feat_df.index[s].date()}")

    cyc_scaled = np.concatenate(blocks, axis=1)[:, :cycle_len]     # (K, L, 30)
    K, L, _ = cyc_scaled.shape
    cyc_ret = scaler.inverse_transform(cyc_scaled.reshape(K * L, NUM_VARS)).reshape(K, L, NUM_VARS)
    if whitener is not None:
        cyc_ret = color_rclose(cyc_ret, colorizer if colorizer is not None else whitener)
    return cyc_ret


def save_cycle_output(cyc_ret, feat_df, origin_idx, cycle_len, k, out_dir, meta):
    """存該期生成結果：多路徑存 .npy，另存 path 0 的 CSV 方便肉眼檢查。"""
    cyc_dates = feat_df.index[origin_idx: origin_idx + cycle_len]
    np.save(os.path.join(out_dir, "cycles", f"cycle_{k:02d}_returns.npy"), cyc_ret)
    df_out = pd.DataFrame(cyc_ret[0], columns=COLS_ORDER)
    df_out.insert(0, "date", cyc_dates)
    df_out.to_csv(os.path.join(out_dir, "cycles", f"cycle_{k:02d}_returns.csv"), index=False)
    with open(os.path.join(out_dir, "cycles", f"cycle_{k:02d}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_cycle_ckpt(args, out_dir, k):
    """--generate_only：載回該期存下的模型、scaler 與 whitener。"""
    ck = os.path.join(out_dir, "ckpt", f"cycle_{k:02d}.pth")
    sc = os.path.join(out_dir, "ckpt", f"cycle_{k:02d}_scaler.pkl")
    if not (os.path.exists(ck) and os.path.exists(sc)):
        raise FileNotFoundError(
            f"找不到 {ck} 或 {sc}\n--generate_only 需要先用 --save_ckpt 跑過一次完整訓練。")
    exp = Exp_Main(args)
    exp.model.load_state_dict(torch.load(ck, map_location=DEVICE))
    exp.model.eval()
    with open(sc, "rb") as f:
        scaler = pickle.load(f)
    wp = os.path.join(out_dir, "ckpt", f"cycle_{k:02d}_whitener.npy")
    whitener = np.load(wp) if os.path.exists(wp) else None
    cp = os.path.join(out_dir, "ckpt", f"cycle_{k:02d}_colorizer.npy")
    colorizer = np.load(cp) if os.path.exists(cp) else None
    gp = os.path.join(out_dir, "ckpt", f"cycle_{k:02d}_garch.npy")
    garch = np.load(gp) if os.path.exists(gp) else None
    return exp, scaler, whitener, garch, colorizer


def run_one_cycle(cli, args, k, origin_idx, cycle_len, feat_df, out_dir):
    """回傳該期生成的 (n_paths, cycle_len, NUM_VARS) log-return（原始尺度）。"""
    origin_date = feat_df.index[origin_idx]
    win_dir = os.path.join(out_dir, "windows")
    os.makedirs(win_dir, exist_ok=True)
    win_csv = os.path.join(win_dir, f"win_{k:02d}.csv")

    # ── 訓練視窗：嚴格 [origin-1260, origin)，不含 origin ──
    win_df = feat_df.iloc[origin_idx - TRAIN_LEN: origin_idx]
    assert len(win_df) == TRAIN_LEN
    assert win_df.index[-1] < origin_date, "訓練視窗洩漏到 origin 或未來！"

    # ── 時變波動縮放：GARCH 只由「當期訓練窗」擬合，不含 origin 與未來 ──
    raw_win = win_df.values.copy()          # 波動標準化前的原始報酬，供著色矩陣估計
    garch = None; vstruct = None
    if getattr(cli, "vol_scaling", False):
        garch = fit_garch(win_df.values)
        vstruct = fit_vol_structure(win_df.values, garch)
        sig, _ = garch_sigma_causal(win_df.values, garch)     # 嚴格因果
        wz = win_df.values.copy()
        wz[:, RCLOSE_IDX] = wz[:, RCLOSE_IDX] / np.maximum(sig, 1e-12)
        win_df = pd.DataFrame(wz, index=win_df.index, columns=win_df.columns)
        ab = garch[:, 1] + garch[:, 2]
        print(f"    [vol] GARCH α+β={ab.mean():.3f}；波動共同成分 rho={vstruct['rho']:.3f}；"
              f"殘差 Student-t df={vstruct['df']:.2f}；標準化後 z std={wz[:, RCLOSE_IDX].std():.3f}")

    # ── Cross-Asset 白化：L 只由「當期訓練窗」估計，不含 origin 與未來 ──
    whitener = None
    colorizer = None
    if getattr(cli, "whiten_cross_asset", False):
        whitener = fit_whitener(win_df.values)
        # 著色矩陣：搭配 --vol_scaling 時，白化是在 z=r/sigma 空間做的，
        # 而 corr(z) 系統性低於 corr(r)（實測落差 0.01~0.13，波動期最大）——
        # 真實報酬的相關有一部分來自**共同波動**（市場動盪時所有資產同步大幅波動），
        # 逐檔標準化會把它移除，而生成時獨立的 sigma 路徑補不回來。
        # 因此著色改用「原始報酬的相關」，把要的最終相關直接注入。
        if getattr(cli, "vol_scaling", False) and raw_win is not None:
            colorizer = fit_whitener_corr(raw_win)
            iu_ = np.triu_indices(N_TICKER, 1)
            c_r = np.corrcoef(raw_win[:, RCLOSE_IDX], rowvar=False)[iu_].mean()
            c_z = np.corrcoef(win_df.values[:, RCLOSE_IDX], rowvar=False)[iu_].mean()
            print(f"    [color] 著色目標 corr(r)={c_r:.3f}（白化空間 corr(z)={c_z:.3f}）")
        win_df = pd.DataFrame(whiten_rclose(win_df.values, whitener),
                              index=win_df.index, columns=win_df.columns)
        print(f"    [whiten] 由訓練窗估 L（rclose 10x10），"
              f"白化後平均跨資產相關 "
              f"{np.corrcoef(win_df.values[:, RCLOSE_IDX], rowvar=False)[np.triu_indices(N_TICKER,1)].mean():+.4f}")

    win_df.reset_index().rename(columns={"index": "date"}).to_csv(win_csv, index=False)

    args.root_path = win_dir + os.sep
    args.data_path = os.path.basename(win_csv)

    print(f"\n  ── 訓練視窗 {win_df.index[0].date()} ~ {win_df.index[-1].date()} "
          f"({TRAIN_LEN} 日) → origin {origin_date.date()}")

    meta_base = {
        "cycle": k,
        "origin_date": str(origin_date.date()),
        "train_window": [str(win_df.index[0].date()), str(win_df.index[-1].date())],
        "cycle_len": int(cycle_len),
        "n_paths": int(cli.n_paths),
        "gen_seed": int(cli.gen_seed),
        "vol_scale": float(cli.vol_scale),
    }

    # ── --generate_only：跳過訓練，直接載回存好的模型 ──
    if cli.generate_only:
        print(f"  [Cycle {k:02d}] generate_only：載入既有 checkpoint，只重新取樣")
        exp, scaler, whitener, garch, colorizer = load_cycle_ckpt(args, out_dir, k)
        cyc_ret = generate_cycle(exp, scaler, cli, k, origin_idx, cycle_len,
                                 feat_df, whitener, garch, colorizer)
        mp = os.path.join(out_dir, "cycles", f"cycle_{k:02d}_meta.json")
        old = json.load(open(mp)) if os.path.exists(mp) else {}
        meta_base["best_step"] = old.get("best_step")
        meta_base["val_history"] = old.get("val_history", [])
        save_cycle_output(cyc_ret, feat_df, origin_idx, cycle_len, k, out_dir, meta_base)
        del exp
        gc.collect()
        torch.cuda.empty_cache()
        return cyc_ret

    # ================= 選模階段 =================
    # --force_step：跳過 Stage A，直接指定 Stage B 的訓練步數。
    # 用途是控制實驗 —— 比較不同 parameterization 時，val-MSE 會為各組選出
    # 不同的 step（x_start 常被選在極早期），使「參數化差異」與「訓練長度差異」
    # 混在一起。固定步數才能把 parameterization 單獨隔離出來。
    if int(getattr(cli, "force_step", 0)) > 0:
        best_step, history = int(cli.force_step), []
        print(f"  [Cycle {k:02d}] Stage A: 跳過（--force_step {best_step}）")
    else:
        best_step, history = None, None
    if best_step is None:
        print(f"  [Cycle {k:02d}] Stage A: 選模 (train={SEL_TRAIN_LEN}, val={SEL_VAL_LEN})")
        set_seed(cli.seed + 1000 * k)
        args.wf_train_len, args.wf_val_len = SEL_TRAIN_LEN, SEL_VAL_LEN
        exp = Exp_Main(args)
        tr_set, tr_loader = exp._get_data(flag="train")
        va_set, va_loader = exp._get_data(flag="val")
        print(f"    train windows={len(tr_set)}  val windows={len(va_set)}  "
              f"steps/epoch={len(tr_loader)}")

        pretrain_dlinear(exp, tr_loader, cli.pretrain_steps)
        best_step, history = train_diffusion(
            exp, tr_loader, max_steps=cli.max_steps,
            val_loader=va_loader, eval_every=cli.eval_every,
            val_batches=cli.val_batches, log_every=cli.log_every)

        del exp, tr_set, tr_loader, va_set, va_loader
        gc.collect()
        torch.cuda.empty_cache()

    # ================= 正式階段 =================
    print(f"  [Cycle {k:02d}] Stage B: 正式訓練 (train={TRAIN_LEN}, 至 step {best_step})")
    set_seed(cli.seed + 1000 * k + 1)            # 重新初始化（不 warm-start）
    args.wf_train_len, args.wf_val_len = TRAIN_LEN, 0
    exp = Exp_Main(args)
    tr_set, tr_loader = exp._get_data(flag="train")
    print(f"    train windows={len(tr_set)} (規格 1177)  steps/epoch={len(tr_loader)}")
    scaler = tr_set.scaler                        # 以完整 1260 日 fit 的 scaler

    pretrain_dlinear(exp, tr_loader, cli.pretrain_steps)
    train_diffusion(exp, tr_loader, max_steps=cli.max_steps,
                    target_steps=best_step, log_every=cli.log_every)

    # ── 存 checkpoint + scaler（供日後 --generate_only 追加路徑）──
    if cli.save_ckpt:
        ck_dir = os.path.join(out_dir, "ckpt")
        os.makedirs(ck_dir, exist_ok=True)
        torch.save(exp.model.state_dict(), os.path.join(ck_dir, f"cycle_{k:02d}.pth"))
        if whitener is not None:
            np.save(os.path.join(ck_dir, f"cycle_{k:02d}_whitener.npy"), whitener)
        if colorizer is not None:
            np.save(os.path.join(ck_dir, f"cycle_{k:02d}_colorizer.npy"), colorizer)
        if garch is not None:
            np.savez(os.path.join(ck_dir, f"cycle_{k:02d}_garch.npz"),
                     par=garch, **{k2: np.asarray(v2) for k2, v2 in vstruct.items()})
        with open(os.path.join(ck_dir, f"cycle_{k:02d}_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        print(f"    [ckpt] 已存 {ck_dir}/cycle_{k:02d}.pth")

    # ================= 生成 21/21/21 × n_paths =================
    cyc_ret = generate_cycle(exp, scaler, cli, k, origin_idx, cycle_len,
                             feat_df, whitener, garch, colorizer)

    meta_base["best_step"] = int(best_step)
    meta_base["max_steps"] = int(cli.max_steps)
    meta_base["val_history"] = history
    save_cycle_output(cyc_ret, feat_df, origin_idx, cycle_len, k, out_dir, meta_base)

    del exp, tr_set, tr_loader
    gc.collect()
    torch.cuda.empty_cache()
    return cyc_ret


# ===========================================================
# Main
# ===========================================================
def main():
    p = argparse.ArgumentParser(description="DOW10 Walk-Forward Generation (2018-2024)")
    p.add_argument("--tag",      type=str, default="dow10_wf")
    p.add_argument("--out_dir",  type=str, default=None)
    p.add_argument("--returns_csv", type=str, default=RETURNS_CSV)
    p.add_argument("--ohlc_csv",    type=str, default=OHLC_CSV)
    # 排程
    p.add_argument("--force_step", type=int, default=0,
                   help="跳過 Stage A 選模，直接以此步數做正式訓練（控制實驗用）")
    p.add_argument("--start_cycle", type=int, default=0, help="從第幾期開始（配合 --resume）")
    p.add_argument("--max_cycles",  type=int, default=0, help="只跑前 N 期（0=全部，除錯用）")
    p.add_argument("--resume", action="store_true", help="已存在的 cycle_XX_returns.csv 直接沿用")
    # 訓練
    p.add_argument("--max_steps",     type=int, default=2000,
                   help="選模階段的 step 上限，同時是兩階段 LR/annealing 的基準")
    p.add_argument("--eval_every",    type=int, default=100, help="每幾 step 算一次 validation MSE")
    p.add_argument("--val_batches",   type=int, default=0, help="驗證只取前 N 個 batch（0=全部）")
    p.add_argument("--pretrain_steps", type=int, default=500, help="DLinear 預訓 step 數")
    p.add_argument("--log_every",     type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=16)
    p.add_argument("--use_amp", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--seed",    type=int, default=2024)
    # 模型（須與想複製的設定一致）
    p.add_argument("--diff_steps",      type=int, default=100)
    p.add_argument("--type_sampler",    type=str, default="dpm", choices=["none", "dpm"])
    p.add_argument("--parameterization", type=str, default="x_start", choices=["x_start", "noise", "v"])
    p.add_argument("--ddpm_clip",       type=float, default=100.0)
    p.add_argument("--noise_dist",      type=str, default="gaussian", choices=["gaussian", "studentt"])
    p.add_argument("--noise_df",        type=float, default=5.0)
    p.add_argument("--use_xt_residual", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--use_window_normalization", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--use_volatility_loss", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--use_leverage_loss",   type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--lambda_vol_max", type=float, default=0.1)
    p.add_argument("--lambda_lev_max", type=float, default=0.1)
    p.add_argument("--vol_scaling", action="store_true",
                   help="時變波動縮放：訓練前以 GARCH(1,1) 把 rclose 標準化成 z，"
                        "生成後由已生成路徑遞迴還原 sigma_t。波動聚集與厚尾成為結構性保證")
    p.add_argument("--vol_common", action="store_true",
                   help="sigma 的驅動加入市場共同成分（ablation 用；實測會降低波動聚集）")
    p.add_argument("--heavy_tail_residual", action="store_true",
                   help="第四層：把生成 z 的邊際保序校準成 Student-t(df)，df 由訓練窗"
                        "標準化殘差的峰度反推。單調變換，完全保留 copula")
    p.add_argument("--whiten_cross_asset", action="store_true",
                   help="Cross-Asset 白化（等價於 correlated noise prior）：訓練前把 rclose "
                        "以訓練窗的 Cholesky 因子白化，生成後著色還原。不動擴散數學")
    p.add_argument("--aux_no_teacher_forcing", action="store_true",
                   help="輔助 loss（stylized / cross-asset）改用 y_clean=None 的額外 forward 計算，\n"
                        "避開 future mixup 造成的 teacher forcing。每步多一次 forward")
    # ── Cross-Asset Dependency Loss ──
    p.add_argument("--lambda_cross_max", type=float, default=0.0,
                   help="cross-asset correlation loss 權重；0 = 關閉（預設），"
                        "ablation 建議 0 / 0.01 / 0.05 / 0.1")
    p.add_argument("--cross_min_t_ratio", type=float, default=0.0,
                   help="只在 t >= ratio*T 套用（下界）。0.0 = 不設限")
    p.add_argument("--cross_gen_aligned", action="store_true",
                   help="Generation-Aligned：改用純 N(0,I) 輸入 + 高噪聲 t 計算 cross loss，"
                        "消除 x_t 中的 x0 殘留（障礙 C）。每步多一次 forward")
    p.add_argument("--cross_gen_t_ratio", type=float, default=0.9,
                   help="generation-aligned 模式下 t 的抽樣下界比例（0.9 -> t in [90,100)）")
    p.add_argument("--cross_timestep_ratio", type=float, default=1.0,
                   help="只在 t < ratio*T 套用。預設 1.0（所有 t），因為生成鏈每一步"
                        "都用 x0_hat，只在低 t 施加約束來不及修正高 t 產生的不相關結構")
    # ── Energy Score（proper scoring rule，直接懲罰 under-dispersion）──
    p.add_argument("--lambda_energy_max", type=float, default=0.0,
                   help="energy score 權重上限；0 = 關閉（預設）")
    p.add_argument("--energy_n_samples", type=int, default=3,
                   help="每個 t 抽幾個 x0 估計來算 energy score（訓練成本 ~線性成長）")
    p.add_argument("--energy_min_t_ratio", type=float, default=0.6,
                   help="只在 t >= ratio*T 套用。低 t 的真實條件分布本來就集中，"
                        "在那裡獎勵分散度會逼出假雜訊")
    p.add_argument("--energy_beta", type=float, default=1.0,
                   help="energy score 的 beta 次方，需在 (0,2)")
    # 生成
    p.add_argument("--vol_scale", type=float, default=1.0,
                   help="振幅校準；既有 DOW10 校準值為 3.86")
    p.add_argument("--n_paths", type=int, default=1,
                   help="一次生成幾份資料（同一批模型、同一組真實 condition，只差取樣噪聲）")
    p.add_argument("--gen_seed", type=int, default=0,
                   help="生成亂數種子偏移；改這個可在同一批模型上追加不同的路徑")
    p.add_argument("--save_ckpt", action="store_true",
                   help="每期存模型與 scaler（約 12MB/期），供日後 --generate_only 追加路徑")
    p.add_argument("--generate_only", action="store_true",
                   help="跳過訓練，載入 --save_ckpt 存下的模型直接取樣")
    p.add_argument("--long_format", action="store_true",
                   help="價格輸出改成長格式 date,tic,close,high,low（每日 10 列，"
                        "檔名 path_XX_long.csv）；不加則維持寬表 path_XX_prices.csv")
    cli = p.parse_args()

    if cli.generate_only:
        cli.resume = False        # 目的就是重新取樣，不能沿用舊的生成結果

    out_dir = str(resolve_output_path(cli.out_dir)) if cli.out_dir else str(walkforward_path(cli.tag))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "cycles"), exist_ok=True)

    # ── 讀資料 ─────────────────────────────────────────────
    if not os.path.exists(cli.returns_csv):
        raise FileNotFoundError(
            f"找不到 {cli.returns_csv}\n請先執行：python data_provider/preprocess_walkforward_dow10.py")
    feat_df = pd.read_csv(cli.returns_csv, parse_dates=["date"]).set_index("date")
    feat_df = feat_df[COLS_ORDER]
    assert COLS_ORDER[-1] == "WMT_rlow", "target 必須是最後一欄，否則 Dataset 會重排欄位順序。"

    ohlc = pd.read_csv(cli.ohlc_csv, parse_dates=["date"]).set_index("date")

    dates  = feat_df.index
    P0     = int(dates.searchsorted(pd.Timestamp(PRED_START)))
    P_end  = int(dates.searchsorted(pd.Timestamp(PRED_END), side="right"))
    n_pred = P_end - P0

    n_cycles   = int(np.ceil(n_pred / STRIDE))
    last_len   = n_pred - STRIDE * (n_cycles - 1)
    origins    = [P0 + STRIDE * k for k in range(n_cycles)]
    cycle_lens = [min(STRIDE, n_pred - STRIDE * k) for k in range(n_cycles)]

    print("=" * 74)
    print("  DOW10 Walk-Forward Generation")
    print(f"  預測期      : {dates[P0].date()} ~ {dates[P_end-1].date()}  共 {n_pred} 個共同交易日")
    print(f"  重訓期數    : {n_cycles} 期（stride={STRIDE}，最後一期 {last_len} 日）")
    print(f"  每期訓練窗  : {TRAIN_LEN} 日 → {TRAIN_LEN - SEQ_LEN - PRED_LEN + 1} 個訓練窗口")
    print(f"  選模        : {SEL_TRAIN_LEN}/{SEL_VAL_LEN}，max_steps={cli.max_steps}, "
          f"eval_every={cli.eval_every}")
    print(f"  輸出        : {out_dir}")
    print("=" * 74)
    if n_pred != 1761:
        print(f"⚠️  實際 {n_pred} 天 != 規格 1761 天；期數與最後一期長度已依實際資料推導。")
    if n_cycles != 28 or last_len != 60:
        print(f"⚠️  期數={n_cycles}（規格 28）、最後一期={last_len} 日（規格 60）。")
    if P0 < TRAIN_LEN:
        raise ValueError(f"第一個 origin 之前只有 {P0} 天，不足 {TRAIN_LEN} 天。請延長前處理的 START_DATE。")

    args = build_args(cli)

    # ── 逐期 walk-forward ──────────────────────────────────
    all_rets = [None] * n_cycles
    end_cycle = n_cycles if not cli.max_cycles else min(n_cycles, cli.start_cycle + cli.max_cycles)
    t_all = time.time()
    for k in range(cli.start_cycle, end_cycle):
        cyc_npy = os.path.join(out_dir, "cycles", f"cycle_{k:02d}_returns.npy")
        cyc_csv = os.path.join(out_dir, "cycles", f"cycle_{k:02d}_returns.csv")
        if cli.resume:
            prev = None
            if os.path.exists(cyc_npy):
                arr = np.load(cyc_npy)
                if arr.shape[0] >= cli.n_paths:
                    prev = arr[:cli.n_paths]
                else:
                    print(f"[Cycle {k:02d}] 既有結果只有 {arr.shape[0]} 條路徑 "
                          f"< 需要的 {cli.n_paths} 條 → 重跑本期")
            elif os.path.exists(cyc_csv) and cli.n_paths == 1:
                prev = pd.read_csv(cyc_csv)[COLS_ORDER].values[np.newaxis]
            elif os.path.exists(cyc_csv):
                print(f"[Cycle {k:02d}] 既有結果是單路徑 CSV < 需要的 {cli.n_paths} 條 → 重跑本期")
            if prev is not None:
                all_rets[k] = prev
                print(f"\n[Cycle {k:02d}/{n_cycles-1}] 沿用既有結果（{prev.shape[0]} 條路徑）")
                continue
        print(f"\n{'='*74}\n[Cycle {k:02d}/{n_cycles-1}] origin={dates[origins[k]].date()}  "
              f"生成 {cycle_lens[k]} 日\n{'='*74}")
        all_rets[k] = run_one_cycle(cli, args, k, origins[k], cycle_lens[k], feat_df, out_dir)
        print(f"[Cycle {k:02d}] 完成，累計耗時 {(time.time()-t_all)/60:.1f} 分鐘")

    if any(r is None for r in all_rets):
        done = sum(r is not None for r in all_rets)
        print(f"\n⚠️  只完成 {done}/{n_cycles} 期（--max_cycles/--start_cycle），"
              f"跳過後續彙整。補齊後加 --resume 重跑即可彙整。")
        return

    # ── 彙整：先按日期串接全部 1,761 日（逐路徑）────────────
    all_ret    = np.concatenate(all_rets, axis=1)      # (K, n_pred, NUM_VARS)
    K          = all_ret.shape[0]
    pred_dates = dates[P0:P_end]
    assert all_ret.shape == (K, n_pred, NUM_VARS)

    # ── 時變波動還原：z -> r ──────────────────────────────
    # cycles/*.npy 存的是 z（模型的原生輸出空間），在此一次遞迴還原，
    # 因此 sigma 的狀態自然跨區塊、跨期連續，且 --resume 也不受影響。
    # 每一段用「該期訓練窗擬合的 GARCH 參數」，與 walk-forward 協定一致；
    # sigma_t 由**已生成**的報酬遞迴，生成過程不接觸任何真實資料。
    if getattr(cli, "vol_scaling", False):
        gpars, gstructs = [], []
        for k in range(n_cycles):
            gp = os.path.join(out_dir, "ckpt", f"cycle_{k:02d}_garch.npz")
            if os.path.exists(gp):
                d_ = np.load(gp)
                gpars.append(d_["par"])
                gstructs.append({kk: d_[kk].item() if d_[kk].ndim == 0 else d_[kk]
                                 for kk in ("rho", "df", "v_m", "v_stat")})
            else:
                gpars.append(None); gstructs.append(None)
        if all(g is None for g in gpars):
            print("⚠️  --vol_scaling 開啟但找不到 GARCH 參數（需 --save_ckpt），跳過還原。")
        else:
            last = next(g for g in gpars if g is not None)
            gpars = [g if g is not None else last for g in gpars]
            lastst = next(g for g in gstructs if g is not None)
            gstructs = [g if g is not None else lastst for g in gstructs]
            # 共同波動因子預設關閉：實測它把 LB Q 從 416 砍到 161
            # （共同驅動讓個股的波動衝擊被平均掉，持續性被稀釋），
            # 且原本要解決的「相關被打散」問題經單元測試證實並不存在。
            # 程式碼保留供 ablation。
            if not getattr(cli, "vol_common", False):
                gstructs = [dict(g, rho=0.0) for g in gstructs]
            # 邊際校準（第四層）：把 z 的邊際換成訓練窗估出的 Student-t(df)。
            # 單調保序，完全保留 copula，因此不影響前面幾層建立的相依結構。
            if getattr(cli, "heavy_tail_residual", False):
                dfm = float(np.median([g["df"] for g in gstructs]))
                for p_ in range(K):
                    all_ret[p_] = calibrate_marginal_t(all_ret[p_], dfm)
                print(f"  [tail] 邊際校準為 Student-t(df={dfm:.2f})（由訓練窗殘差峰度反推）")
            # 初始變異數 = 第 0 期訓練窗尾端的因果 sigma^2（唯一用到真實資料之處，
            # 等同於價格重建的 seed，之後完全由生成路徑自我推進）
            _, v0 = garch_sigma_causal(feat_df.values[origins[0] - TRAIN_LEN: origins[0]],
                                       gpars[0])
            for p_ in range(K):
                v = v0.copy()
                off = 0
                for k in range(n_cycles):
                    ln = cycle_lens[k]
                    seg, v = descale_recursive(all_ret[p_, off:off + ln], gpars[k], v,
                                               struct=gstructs[k])
                    all_ret[p_, off:off + ln] = seg
                    off += ln
            sd_ = all_ret[:, :, RCLOSE_IDX].std()
            print(f"  [vol] 已由生成路徑遞迴還原 sigma_t（{K} 條 × {n_pred} 日），"
                  f"還原後 rclose std = {sd_:.5f}")

    # ── 價格重建：唯一 seed = 2017 最後一個共同交易日的真實 Close ──
    seed_date  = dates[P0 - 1]
    seed_close = ohlc.loc[seed_date, [f"{t}_Close" for t in TICKERS]].values.astype(np.float64)
    print(f"\n  價格 seed：{seed_date.date()}  "
          f"{ {t: round(v,2) for t, v in zip(TICKERS, seed_close)} }")

    paths_dir  = os.path.join(out_dir, "paths")
    price_dir  = os.path.join(paths_dir, "price")     # 價格（OHLC）
    return_dir = os.path.join(paths_dir, "returns")   # 原始 log-return
    os.makedirs(price_dir, exist_ok=True)
    os.makedirs(return_dir, exist_ok=True)
    real_rclose = feat_df.iloc[P0:P_end][[f"{t}_rclose" for t in TICKERS]].values
    real_m      = compute_stylized_facts(real_rclose)

    syn_closes, summary = [], []
    for i in range(K):
        ret_i = all_ret[i]
        df_ret = pd.DataFrame(ret_i, columns=COLS_ORDER)
        df_ret.insert(0, "date", pred_dates)
        df_ret.to_csv(os.path.join(return_dir, f"path_{i:02d}_returns.csv"), index=False)

        c, h, l = reconstruct_ohlc(ret_i, seed_close)
        syn_closes.append(c)

        if cli.long_format:      # date,tic,close,high,low
            df_price = to_long_format(pred_dates, c, h, l)
            # 價格檔用 1-based 三位數（path_001.csv…），與既有 paths/price/ 命名一致；
            # returns 檔沿用 0-based 兩位數，所以 path_001.csv 對應 path_00_returns.csv。
            price_name = f"path_{i+1:03d}.csv"
            legacy_name = "wf_generated_prices_long.csv"
        else:                    # 寬表：date + {ticker}_Close/High/Low
            price_cols = {}
            for j, t in enumerate(TICKERS):
                price_cols[f"{t}_Close"] = c[:, j]
                price_cols[f"{t}_High"]  = h[:, j]
                price_cols[f"{t}_Low"]   = l[:, j]
            df_price = pd.DataFrame(price_cols)
            df_price.insert(0, "date", pred_dates)
            price_name = f"path_{i+1:03d}_prices.csv"
            legacy_name = "wf_generated_prices.csv"
        df_price.to_csv(os.path.join(price_dir, price_name), index=False)

        if i == 0:   # 同時寫出單路徑時的檔名，維持既有下游腳本相容
            df_ret.to_csv(os.path.join(out_dir, "wf_generated_returns.csv"), index=False)
            df_price.to_csv(os.path.join(out_dir, legacy_name), index=False)

        m = compute_stylized_facts(ret_i[:, RCLOSE_IDX])
        vr = float(np.mean(ret_i[:, RCLOSE_IDX].std(axis=0) / real_rclose.std(axis=0)))
        # price_file / returns_file：path 欄位是 0-based，但價格檔是 1-based 三位數，
        # 直接寫出檔名免得對照時被這個偏移絆到。
        summary.append({"path": i,
                        "price_file": price_name,
                        "returns_file": f"path_{i:02d}_returns.csv",
                        "vol_ratio": vr,
                        **{k2: float(m["average"][k2]) for k2 in
                           ["tail_index", "skewness", "jump_diffusion",
                            "vol_cluster_acf1", "leverage_effect"]}})
        if i == 0:
            syn_m = m

    print(f"✅ {K} 份資料已儲存：{return_dir}/path_XX_returns.csv 與 "
          f"{price_dir}/"
          f"{'path_XXX.csv（date,tic,close,high,low）' if cli.long_format else 'path_XXX_prices.csv'}")

    # ── 各路徑的統計摘要（含跨路徑的離散度）──
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(os.path.join(out_dir, "wf_paths_summary.csv"), index=False)
    real_row = {"vol_ratio": 1.0,
                **{k2: float(real_m["average"][k2]) for k2 in
                   ["tail_index", "skewness", "jump_diffusion",
                    "vol_cluster_acf1", "leverage_effect"]}}
    print(f"\n{'metric':<20}{'real':>10}{'paths mean':>12}{'paths std':>11}{'min':>10}{'max':>10}")
    print("-" * 73)
    for k2 in ["vol_ratio", "tail_index", "skewness", "jump_diffusion",
               "vol_cluster_acf1", "leverage_effect"]:
        v = df_sum[k2].values
        print(f"{k2:<20}{real_row[k2]:>10.4f}{v.mean():>12.4f}{v.std():>11.4f}"
              f"{v.min():>10.4f}{v.max():>10.4f}")
    print(f"（跨路徑 std 只含取樣噪聲；模型隨機性要靠不同 --seed 的完整重跑才會進來）")

    # ── 繪圖（path 0 為主，其餘路徑淡色疊上以看離散度）──
    real_close = ohlc.loc[pred_dates, [f"{t}_Close" for t in TICKERS]].values
    plot_prices(pred_dates, syn_closes[0], real_close,
                [dates[o] for o in origins],
                os.path.join(out_dir, "wf_price.png"),
                extra_closes=syn_closes[1:10])

    # ── Stylized Facts（path 0 的完整報表；重用 generate_dow10 的實作）──
    print("\n計算 Stylized Facts（path 0）...")
    meta = {
        "tag": cli.tag,
        "period": f"{pred_dates[0].date()} ~ {pred_dates[-1].date()}",
        "n_days": n_pred,
        "n_cycles": n_cycles,
        "stride": STRIDE,
        "train_len": TRAIN_LEN,
        "select_split": f"{SEL_TRAIN_LEN}/{SEL_VAL_LEN}",
        "max_steps": cli.max_steps,
        "vol_scale": cli.vol_scale,
        "n_paths": K,
        "gen_seed": cli.gen_seed,
        "note": "本報表為 path 0；各路徑摘要見 wf_paths_summary.csv",
        "strategy": "Walk-Forward retrain, real-cond blocks, chained prices (no reset)",
    }
    print_and_save_stylized_facts(
        real_m, syn_m, os.path.join(out_dir, "wf_stylized_facts.txt"), meta)

    best_steps = []
    for k in range(n_cycles):
        mp = os.path.join(out_dir, "cycles", f"cycle_{k:02d}_meta.json")
        if os.path.exists(mp):
            # --generate_only 寫到新的 out_dir 時沒有前次的 meta 可繼承，
            # best_step 會是 None；直接丟進 np.mean 會 TypeError。
            bs = json.load(open(mp)).get("best_step")
            if bs is not None:
                best_steps.append(bs)
    if best_steps:
        print(f"\n各期選定 step：{best_steps}")
        print(f"  mean={np.mean(best_steps):.0f}  min={min(best_steps)}  max={max(best_steps)}")

    print(f"\n🎉 完成！總耗時 {(time.time()-t_all)/60:.1f} 分鐘，結果於 {out_dir}")


if __name__ == "__main__":
    main()
