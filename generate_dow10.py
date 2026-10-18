"""
generate_dow10.py
==================
DOW10 實驗：生成 2024 全年合成金融時間序列。

Rolling 策略（Pure Real Condition）：
  - 每次生成的 Condition 永遠來自真實資料（2023 尾端 + 2024 真實）
  - Stride = PRED_LEN = 21 天
  - Roll i: cond = real_data[i*21 : i*21+63]  →  gen[i*21 : i*21+21]
  - 12 rolls × 21 天 = 252 天 ≈ 2024 全年

價格還原策略（Chained Generated Returns）：
  - 所有生成 return 串接，前一天用生成結果還原（不用真實前一天）
  - 避免 block 交界處價格跳躍

現有實驗完全不受影響（全新獨立檔案）。
"""

import os
import sys
import argparse
import random
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.abspath("."))
from exp.exp_main import Exp_Main

warnings.filterwarnings("ignore")

# ===========================================================
# 全域常數
# ===========================================================
TICKERS   = ["AAPL", "AMGN", "CRM", "CSCO", "IBM",
             "INTC", "MSFT", "NKE", "VZ", "WMT"]
N_TICKER  = len(TICKERS)   # 10
N_FEAT    = 3               # rclose, rhigh, rlow
NUM_VARS  = N_TICKER * N_FEAT  # 30 features

SEQ_LEN   = 63     # condition 視窗長度
LABEL_LEN = 21     # label token
PRED_LEN  = 21     # 每次生成天數
N_ROLLS   = 12     # 12 × 21 = 252 ≈ 2024 全年

DATA_DIR  = "./datasets/prediction/data_finance"
OUT_DIR   = "./outputs/dow10/single_shot/dow10_2024"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# rclose 的 channel index（每 3 個 channel 一組，index 0, 3, 6, ...）
RCLOSE_IDX = [i * 3 for i in range(N_TICKER)]  # [0,3,6,9,12,15,18,21,24,27]

# ===========================================================
# 1. 建立 args（必須與訓練時一致）
# ===========================================================
def build_args(tag="dow10", use_window_normalization=False, use_xt_residual=False,
               parameterization="x_start", ddpm_clip=100.0,
               noise_dist="gaussian", noise_df=5.0):
    class Args: pass
    args = Args()
    args.tag = tag
    args._ddpm_clip_override = ddpm_clip
    args.noise_dist = noise_dist   # 須與訓練一致（Student-t 取樣需開）
    args.noise_df   = noise_df
    args.use_xt_residual = use_xt_residual   # 需與訓練時一致，否則 checkpoint 載入失敗
    args._parameterization_override = parameterization  # 見下方覆寫

    args.model            = "DDPM"
    args.use_window_normalization = use_window_normalization
    args.stage_mode       = "TWO"
    args.is_training      = 0
    args.out_figures      = 0
    args.vis_ar_part      = 0
    args.vis_MTS_analysis = 0
    args.sample_times     = 1
    args.beta_dist_alpha  = -1
    args.our_ddpm_clip    = args._ddpm_clip_override

    args.seq_len   = SEQ_LEN
    args.label_len = LABEL_LEN
    args.pred_len  = PRED_LEN

    args.dataset_name = "custom"
    args.data         = "custom"
    args.weather_type = "mintemp"
    args.root_path    = DATA_DIR + "/"
    args.data_path    = "dow10_returns.csv"
    args.features     = "M"
    args.target       = "WMT_rlow"
    args.num_vars     = NUM_VARS
    args.checkpoints  = "./checkpoints/"
    args.freq         = "b"

    args.interval         = 1000
    args.ot_ode           = True
    args.beta_max         = 0.3
    args.t0               = 1e-4
    args.T                = 1.0
    args.nfe              = 100
    args.dim_LSTM         = 64
    args.diff_steps       = 100
    args.UNet_Type        = "CNN"
    args.D3PM_kernel_size = 5
    args.use_freq_enhance = 0
    args.type_sampler     = "none"
    args.parameterization = args._parameterization_override   # x_start 或 noise，須與訓練一致

    args.model_channels         = 256
    args.ddpm_inp_embed         = 256
    args.ddpm_dim_diff_steps    = 256
    args.ddpm_channels_conv     = 256
    args.ddpm_channels_fusion_I = 256
    args.ddpm_layers_inp        = 5
    args.ddpm_layers_I          = 5
    args.ddpm_layers_II         = 5
    args.cond_ddpm_num_layers   = 5
    args.cond_ddpm_channels_conv = 64

    args.ablation_study_case         = "none"
    args.weight_pred_loss            = 0.0
    args.ablation_study_F_type       = "CNN"
    args.ablation_study_masking_type = "none"
    args.ablation_study_masking_tau  = 0.9

    args.learning_rate   = 1e-3
    args.embed           = "timeF"
    args.use_gpu         = torch.cuda.is_available()
    args.gpu             = 0
    args.use_multi_gpu   = False
    args.devices         = "0"
    args.seed            = 2024
    args.num_workers     = 0
    args.itr             = 1
    args.batch_size      = 32
    args.test_batch_size = 16
    args.use_amp         = True
    args.use_valset      = True

    args.use_volatility_loss = False
    args.use_leverage_loss   = False

    args.model_id = f"custom_{SEQ_LEN}_{PRED_LEN}"
    args.device   = DEVICE
    return args


# ===========================================================
# 2. 載入模型
# ===========================================================
def load_model(args):
    exp = Exp_Main(args)
    setting = (
        f"{args.model_id}_{args.model}_{args.data}"
        f"_ft{args.features}_sl{args.seq_len}_ll{args.label_len}"
        f"_pl{args.pred_len}_dt0_{args.stage_mode}_{args.tag}"
    )
    ckpt_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到 checkpoint: {ckpt_path}\n"
            "請先執行訓練 (run_dow10.sh)"
        )
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    exp.model.eval()
    print(f"✅ 成功載入 checkpoint: {ckpt_path}")
    return exp


# ===========================================================
# 3. 單步推論：給定 (SEQ_LEN, NUM_VARS) 的 condition，生成 (PRED_LEN, NUM_VARS)
# ===========================================================
def generate_one_step(exp, cond_scaled_np):
    """
    cond_scaled_np: (SEQ_LEN, NUM_VARS) — 已標準化的真實資料
    回傳: (PRED_LEN, NUM_VARS) — 標準化空間的生成結果
    """
    x_enc = torch.tensor(cond_scaled_np[np.newaxis], dtype=torch.float32).to(DEVICE)

    label_part = cond_scaled_np[-LABEL_LEN:]
    pad        = np.zeros((PRED_LEN, NUM_VARS), dtype=np.float32)
    dec_inp    = np.concatenate([label_part, pad], axis=0)
    x_dec      = torch.tensor(dec_inp[np.newaxis], dtype=torch.float32).to(DEVICE)

    x_mark_enc = torch.zeros(1, SEQ_LEN,            4, dtype=torch.float32).to(DEVICE)
    x_mark_dec = torch.zeros(1, LABEL_LEN + PRED_LEN, 4, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs, _, _, _, _ = exp.model.forward(
            x_enc, x_mark_enc, x_dec, x_mark_dec,
            sample_times=1
        )

    if outputs.dim() == 4:
        pred = outputs[0][0].cpu().numpy()   # (PRED_LEN, NUM_VARS)
    else:
        pred = outputs[0].cpu().numpy()

    return pred   # (PRED_LEN, NUM_VARS)


# ===========================================================
# 4. Pure Real Condition Rolling Generation
# ===========================================================
def rolling_real_condition_generate(exp, real_matrix_scaled, scaler, vol_scale_factor=1.0):
    """
    real_matrix_scaled: (SEQ_LEN + N_ROLLS*PRED_LEN, NUM_VARS)
        index 0        ~ SEQ_LEN-1          : 2023 最後 SEQ_LEN 個交易日（已標準化）
        index SEQ_LEN  ~ SEQ_LEN+N_ROLLS*21 : 2024 全年真實資料（已標準化）

    Rolling 策略：
        Roll i：cond = real_matrix_scaled[i*PRED_LEN : i*PRED_LEN + SEQ_LEN]
        → 生成 PRED_LEN 天
        → condition 永遠是真實資料，stride = PRED_LEN

    回傳：all_pred (N_ROLLS*PRED_LEN, NUM_VARS) — 反標準化後的 log-return
    """
    all_chunks = []

    for roll in range(N_ROLLS):
        start = roll * PRED_LEN
        end   = start + SEQ_LEN
        cond  = real_matrix_scaled[start:end]   # (63, 30) 真實資料

        print(f"  [Roll {roll+1}/{N_ROLLS}] cond window: index [{start}:{end}]", end="")

        pred_scaled = generate_one_step(exp, cond)  # (21, 30)

        # Volatility Rescaling（選用）
        if vol_scale_factor != 1.0:
            chan_mean  = pred_scaled.mean(axis=0, keepdims=True)
            residual   = pred_scaled - chan_mean
            pred_scaled = chan_mean + residual * vol_scale_factor

        all_chunks.append(pred_scaled)
        print(f"  → gen shape: {pred_scaled.shape}")

    all_pred_scaled = np.concatenate(all_chunks, axis=0)  # (252, 30)
    all_pred        = scaler.inverse_transform(all_pred_scaled)  # raw log-returns
    return all_pred


# ===========================================================
# 5. 反推絕對收盤價（Chained Generated Returns）
# ===========================================================
def reconstruct_prices(all_pred, seed_close):
    """
    all_pred  : (T, NUM_VARS) — 生成的 log-return
    seed_close: (N_TICKER,)  — 2023-12-31 的收盤價

    rclose_idx = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    price[t+1] = price[t] × exp(rclose[t])

    使用「生成的前一天」還原，不用真實前一天 → 避免 block 交界跳躍。
    """
    T = all_pred.shape[0]
    close_prices = np.zeros((T + 1, N_TICKER))
    close_prices[0] = seed_close

    for t in range(T):
        rclose = all_pred[t, RCLOSE_IDX]
        close_prices[t + 1] = close_prices[t] * np.exp(rclose)

    return close_prices   # (T+1, N_TICKER)


# ===========================================================
# 6. 繪圖
# ===========================================================
def plot_prices(close_prices, real_prices=None, title="DOW10 Synthetic vs Real (2024)"):
    T      = close_prices.shape[0]
    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
    axes = axes.flatten()

    for i, (ax, ticker) in enumerate(zip(axes, TICKERS)):
        ax.plot(np.arange(T), close_prices[:, i],
                linewidth=1.5, label="Synthetic", color="tab:orange")
        if real_prices is not None:
            T_real = real_prices.shape[0]
            ax.plot(np.arange(T_real), real_prices[:, i],
                    linewidth=1.5, label="Real", color="tab:blue", alpha=0.8)
        ax.set_title(ticker, fontsize=10)
        ax.set_ylabel("Price (USD)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "dow10_2024_price.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ 股價走勢圖已儲存：{out_path}")


# ===========================================================
# 7. Stylized Facts
# ===========================================================
def hill_tail_index(r, q=0.95):
    x = np.abs(r[r < 0])
    if len(x) < 10:
        return np.nan
    u    = np.quantile(x, q)
    tail = x[x > u]
    if len(tail) < 2:
        return np.nan
    return 1.0 / np.mean(np.log(tail / u))


def jump_measure(r):
    """
    Jump Diffusion 指標：Barndorff-Nielsen & Shephard 相對跳躍測度 (Relative Jump)。
        RV = Σ r_t²                               (realized variance，對跳躍不穩健)
        BV = (π/2) Σ_{t=2..n} |r_t||r_{t-1}|      (bipower variation，對跳躍穩健)
        RJ = max(0, (RV − BV) / RV) ∈ [0, 1)
    值越大代表報酬中「不連續跳躍」成分越多。連續擴散極限下 RV≈BV → RJ≈0。
    """
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return np.nan
    rv = np.sum(r ** 2)
    bv = (np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    if rv < 1e-12:
        return np.nan
    return max(0.0, (rv - bv) / rv)


def compute_stylized_facts(rclose_mat):
    """rclose_mat: (T, N_TICKER)"""
    metrics = {"per_ticker": {}, "average": {}}
    # kurt_list  # Excess Kurtosis 已停用（先註解掉）
    tail_list, acf_list, lev_list, skew_list, jump_list = [], [], [], [], []

    for i, ticker in enumerate(TICKERS):
        r = rclose_mat[:, i]
        r = r[np.isfinite(r)]

        # kurt = stats.kurtosis(r, fisher=True)   # Excess Kurtosis 停用
        # kurt_list.append(kurt)

        tail = hill_tail_index(r)                  # Heavy Tails
        if not np.isnan(tail):
            tail_list.append(tail)

        acf1 = np.corrcoef(r[:-1] ** 2, r[1:] ** 2)[0, 1] if len(r) > 2 else np.nan
        if not np.isnan(acf1):                     # Volatility Clustering
            acf_list.append(acf1)

        lev = np.corrcoef(r[:-1], np.abs(r[1:]))[0, 1] if len(r) > 2 else np.nan
        if not np.isnan(lev):                      # Leverage Effect
            lev_list.append(lev)

        skew = stats.skew(r)                        # Negative Skewness
        skew_list.append(skew)

        jump = jump_measure(r)                      # Jump Diffusion（新增）
        if not np.isnan(jump):
            jump_list.append(jump)

        metrics["per_ticker"][ticker] = {
            # "excess_kurtosis":  kurt,             # 停用
            "tail_index":       tail,
            "skewness":         skew,
            "jump_diffusion":   jump,
            "vol_cluster_acf1": acf1,
            "leverage_effect":  lev,
        }

    # metrics["average"]["excess_kurtosis"] = np.nanmean(kurt_list)   # 停用
    metrics["average"]["tail_index"]       = np.nanmean(tail_list)
    metrics["average"]["skewness"]         = np.nanmean(skew_list)
    metrics["average"]["jump_diffusion"]   = np.nanmean(jump_list)
    metrics["average"]["vol_cluster_acf1"] = np.nanmean(acf_list)
    metrics["average"]["leverage_effect"]  = np.nanmean(lev_list)
    return metrics


def pct_diff(a, b):
    return "N/A" if abs(a) < 1e-8 else f"{(b - a) / abs(a) * 100:+.1f}%"


def print_and_save_stylized_facts(real_m, syn_m, out_path, meta):
    metrics_keys = [
        # ("Excess Kurtosis",     "excess_kurtosis"),   # 停用（先註解掉）
        ("Heavy Tails (Hill)",    "tail_index"),         # 1. Heavy Tails
        ("Negative Skewness",     "skewness"),           # 2. Negative Skewness
        ("Jump Diffusion (RJ)",   "jump_diffusion"),     # 3. Jump Diffusion（新增）
        ("Volatility Clustering", "vol_cluster_acf1"),   # 4. Volatility Clustering
        ("Leverage Effect",       "leverage_effect"),    # 5. Leverage Effect
    ]

    import io, contextlib
    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):
        for ticker in TICKERS:
            print(f"\n{'=' * 68}")
            print(f"  {ticker}")
            print(f"{'=' * 68}")
            print(f"{'Metric':<28} {'Real':>10} {'Synthetic':>10} {'Delta%':>10}  Result")
            print("-" * 68)
            for display_name, key in metrics_keys:
                rv = float(real_m["per_ticker"][ticker][key])
                sv = float(syn_m["per_ticker"][ticker][key])
                flag = "PASS" if abs(rv - sv) / (abs(rv) + 1e-8) < 0.4 else "FAIL"
                print(f"{display_name:<28} {rv:>10.4f} {sv:>10.4f} {pct_diff(rv, sv):>9}  {flag}")

        print(f"\n{'=' * 68}")
        print("  Average (All Tickers)")
        print(f"{'=' * 68}")
        print(f"{'Metric':<28} {'Real':>10} {'Synthetic':>10} {'Delta%':>10}  Result")
        print("-" * 68)
        for display_name, key in metrics_keys:
            rv = float(real_m["average"][key])
            sv = float(syn_m["average"][key])
            flag = "PASS" if abs(rv - sv) / (abs(rv) + 1e-8) < 0.4 else "FAIL"
            print(f"{display_name:<28} {rv:>10.4f} {sv:>10.4f} {pct_diff(rv, sv):>9}  {flag}")
        print(f"{'=' * 68}")

    text = buf.getvalue()
    print(text)   # 終端機顯示

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== DOW10 Stylized Facts Report ===\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
        f.write("=" * 68 + "\n")
        f.write(text)
    print(f"✅ Stylized Facts 已存檔：{out_path}")


# ===========================================================
# Main
# ===========================================================
def main():
    global N_ROLLS, OUT_DIR
    parser = argparse.ArgumentParser(description="DOW10 2024 Rolling Generation")
    parser.add_argument("--tag",      type=str,   default="dow10")
    parser.add_argument("--out_dir",  type=str,   default=None)
    parser.add_argument("--n_rolls",  type=int,   default=12,
                        help="Rolling 次數 (預設 12)")
    parser.add_argument("--hist_end",  type=str,  default="2023-12-31",
                        help="訓練資料結束日（同時作為生成起點 seed）")
    parser.add_argument("--real_start", type=str, default="2024-01-01",
                        help="Stylized Facts 比較的真實資料起始日")
    parser.add_argument("--real_end",   type=str, default="2024-12-31",
                        help="Stylized Facts 比較的真實資料結束日（同時是生成目標年終）")
    parser.add_argument("--use_window_normalization",
                        type=lambda x: x.lower() != "false", default=False)
    parser.add_argument("--vol_scale", type=float, default=1.0)
    parser.add_argument("--use_xt_residual",
                        type=lambda x: x.lower() == "true", default=False,
                        help="須與訓練時一致（含 x_t 殘差 skip 的 checkpoint）")
    parser.add_argument("--parameterization", type=str, default="x_start",
                        choices=["x_start", "noise"],
                        help="須與訓練時一致（ε-param checkpoint 用 noise）")
    parser.add_argument("--ddpm_clip", type=float, default=100.0,
                        help="x0 反推的 clip 範圍（標準化空間；ε-param 建議 3~5）")
    parser.add_argument("--noise_dist", type=str, default="gaussian",
                        choices=["gaussian", "studentt"],
                        help="須與訓練一致：Student-t 重尾先驗需開")
    parser.add_argument("--noise_df", type=float, default=5.0,
                        help="Student-t 自由度（須與訓練一致）")
    cli = parser.parse_args()

    if cli.n_rolls:
        N_ROLLS = cli.n_rolls

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    OUT_DIR = cli.out_dir if cli.out_dir else f"./outputs/dow10/single_shot/{cli.tag}_2024"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  DOW10 Rolling Generation (Pure Real Condition)")
    print(f"  Tag      : {cli.tag}")
    print(f"  Strategy : cond=Real, stride={PRED_LEN}, rolls={N_ROLLS}")
    print(f"  Output   : {OUT_DIR}")
    print("=" * 60)

    # ── 1. 載入模型 ──────────────────────────────────────────────
    args = build_args(cli.tag, use_window_normalization=cli.use_window_normalization,
                      use_xt_residual=cli.use_xt_residual,
                      parameterization=cli.parameterization,
                      ddpm_clip=cli.ddpm_clip,
                      noise_dist=cli.noise_dist, noise_df=cli.noise_df)
    exp  = load_model(args)

    # ── 2. 取得 Scaler（由 train set 決定） ──────────────────────
    train_data, _ = exp._get_data(flag="train")
    scaler = train_data.scaler

    # ── 3. 下載真實資料 ──────────────────────────────────────────
    # 需要：2023 最後 SEQ_LEN 個交易日 + 2024 全年（N_ROLLS*PRED_LEN 天）
    # 保守起見下載 2023-01-01 起的資料，再取最後所需天數
    import yfinance as yf
    print(f"\n下載真實歷史資料（2023 起，確保有足夠 {SEQ_LEN} 天的 2023 尾端及完整 2024）...")
    download_end = (pd.to_datetime(cli.real_end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    raw = yf.download(TICKERS, start="2023-01-01", end=download_end,
                      auto_adjust=True, progress=False)

    adj_close = raw["Close"][TICKERS]
    adj_high  = raw["High"][TICKERS]
    adj_low   = raw["Low"][TICKERS]

    r_close = np.log(adj_close / adj_close.shift(1))
    r_high  = np.log(adj_high  / adj_close)
    r_low   = np.log(adj_low   / adj_close)

    cols_order = []
    for t in TICKERS:
        cols_order += [f"{t}_rclose", f"{t}_rhigh", f"{t}_rlow"]

    feat_df = pd.concat(
        {"rclose": r_close, "rhigh": r_high, "rlow": r_low}, axis=1
    )
    feat_df.columns = [f"{t}_{feat}" for feat, t in feat_df.columns]
    feat_df = feat_df[cols_order].dropna()

    # 切分：2023 尾段 vs 2024
    hist_end_dt  = pd.to_datetime(cli.hist_end)
    real_start_dt = pd.to_datetime(cli.real_start)

    feat_2023 = feat_df[feat_df.index <= hist_end_dt]   # 2023（含）之前
    feat_2024 = feat_df[feat_df.index >= real_start_dt]  # 2024 全年

    if len(feat_2023) < SEQ_LEN:
        raise ValueError(f"2023 資料不足 {SEQ_LEN} 天，只有 {len(feat_2023)} 天。")

    n_2024 = N_ROLLS * PRED_LEN  # 需要的 2024 天數
    if len(feat_2024) < n_2024:
        print(f"⚠️  2024 可用天數 {len(feat_2024)} < 預期 {n_2024}，N_ROLLS 調整為 {len(feat_2024)//PRED_LEN}")
        N_ROLLS = len(feat_2024) // PRED_LEN

    # 合併：2023 尾端 SEQ_LEN 天 + 2024 全年
    tail_2023  = feat_2023.values[-SEQ_LEN:]         # (63, 30)
    data_2024  = feat_2024.values[:N_ROLLS*PRED_LEN]  # (252, 30)
    real_matrix = np.concatenate([tail_2023, data_2024], axis=0).astype(np.float32)
    # real_matrix shape: (63 + 252, 30) = (315, 30)

    print(f"  2023 尾端: {feat_2023.index[-SEQ_LEN].date()} ~ {feat_2023.index[-1].date()}")
    print(f"  2024 資料: {feat_2024.index[0].date()} ~ {feat_2024.index[N_ROLLS*PRED_LEN-1].date()}")
    print(f"  real_matrix shape: {real_matrix.shape}")

    real_matrix_scaled = scaler.transform(real_matrix)

    # Seed close price：2023-12-31 的收盤價
    seed_close = adj_close.loc[adj_close.index <= hist_end_dt].iloc[-1].values.astype(np.float64)
    print(f"  Seed close (2023-12-31): { {t: round(v,2) for t,v in zip(TICKERS, seed_close)} }")

    # ── 4. Rolling Generation ────────────────────────────────────
    print(f"\n開始 Rolling Generation ({N_ROLLS} rolls × {PRED_LEN} days = {N_ROLLS*PRED_LEN} days)...")
    all_pred = rolling_real_condition_generate(
        exp, real_matrix_scaled, scaler, vol_scale_factor=cli.vol_scale
    )   # (252, 30)
    print(f"生成完畢，shape: {all_pred.shape}")

    # ── 5. 存 CSV ────────────────────────────────────────────────
    pred_csv = os.path.join(OUT_DIR, "dow10_generated_returns_2024.csv")
    pd.DataFrame(all_pred, columns=cols_order).to_csv(pred_csv, index=False)
    print(f"✅ 生成 log-return 已儲存：{pred_csv}")

    # ── 6. 反推絕對股價（Chained Generated Returns） ─────────────
    close_prices = reconstruct_prices(all_pred, seed_close)   # (253, 10)
    price_csv = os.path.join(OUT_DIR, "dow10_generated_prices_2024.csv")
    pd.DataFrame(
        close_prices[1:], columns=[f"{t}_Close" for t in TICKERS]
    ).to_csv(price_csv, index=False)
    print(f"✅ 生成股價已儲存：{price_csv}")

    # ── 7. 真實 2024 年股價（用於繪圖比較） ─────────────────────
    real_close_2024 = adj_close.loc[adj_close.index >= real_start_dt].values[:N_ROLLS*PRED_LEN]

    # ── 8. 繪圖 ──────────────────────────────────────────────────
    plot_prices(close_prices[1:], real_prices=real_close_2024)

    # ── 9. Stylized Facts ────────────────────────────────────────
    print("\n計算 Stylized Facts ...")
    syn_rclose_cols = [f"{t}_rclose" for t in TICKERS]
    syn_rclose  = pd.DataFrame(all_pred, columns=cols_order)[syn_rclose_cols].values
    syn_m       = compute_stylized_facts(syn_rclose)

    real_rclose_df = feat_2024[[f"{t}_rclose" for t in TICKERS]].iloc[:N_ROLLS*PRED_LEN]
    real_m         = compute_stylized_facts(real_rclose_df.values)

    sf_path = os.path.join(OUT_DIR, "dow10_stylized_facts_2024.txt")
    meta = {
        "tag":        cli.tag,
        "pred_len":   PRED_LEN,
        "seq_len":    SEQ_LEN,
        "n_rolls":    N_ROLLS,
        "hist_end":   cli.hist_end,
        "real_start": cli.real_start,
        "real_end":   cli.real_end,
        "vol_scale":  cli.vol_scale,
        "strategy":   "Pure Real Condition Rolling",
    }
    print_and_save_stylized_facts(real_m, syn_m, sf_path, meta)

    print(f"\n🎉 完成！所有結果已存放於: {OUT_DIR}")


if __name__ == "__main__":
    main()
