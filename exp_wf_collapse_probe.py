"""
exp_wf_collapse_probe.py
========================
把既有的兩個診斷接到 **walk-forward checkpoint** 上重跑。

背景
----
先前所有 condition-collapse / denoiser-contraction 的證據，都是在「單次 2024 協定」
的模型上取得的（`exp_condition_collapse.py`、`exp_denoiser_sensitivity.py`）。
walk-forward 協定（每 63 日重訓、1260 日訓練窗、28 期）從未測過。

這支腳本回答一個問題：**換了 rolling protocol 之後，collapse 還在嗎？**
答案決定下一步該動 loss 還是先解決隨機性：

  seed_corr 仍 ~0.999  → 模型是確定性回歸器，任何分布層面的干預（stylized loss /
                         condition 加統計量）都無效，要先換目標函數。
  seed_corr 明顯下降   → 前提改變，stylized loss 值得重試。

不修改任何既有檔案；只讀 outputs/<tag>/ckpt/cycle_XX.pth。

用法
----
  python exp_wf_collapse_probe.py --tag dow10_wf_v2 --cycles 0 13 27
"""
import os
import sys
import json
import pickle
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
import walk_forward_dow10 as WF
from exp.exp_main import Exp_Main

DEVICE = WF.DEVICE
RCLOSE_IDX = WF.RCLOSE_IDX
SEQ_LEN, LABEL_LEN, PRED_LEN, NUM_VARS = WF.SEQ_LEN, WF.LABEL_LEN, WF.PRED_LEN, WF.NUM_VARS


def default_cli(tag):
    """build_args() 需要的 cli 屬性。數值與 walk_forward_dow10.main() 的 argparse 預設一致。"""
    # 這些值必須與 walk_forward_dow10.main() 的 argparse 預設**逐一相符**，
    # 否則取樣路徑會與產生 checkpoint 時不同，診斷結果沒有意義。
    # 特別是 type_sampler：預設是 dpm（DPM-Solver），不是 ancestral。
    return SimpleNamespace(
        tag=tag, ddpm_clip=100.0, diff_steps=100,
        type_sampler="dpm", parameterization="x_start",
        use_xt_residual=False, noise_dist="gaussian", noise_df=5.0,
        use_window_normalization=False,
        use_volatility_loss=True, use_leverage_loss=True,
        lambda_vol_max=0.1, lambda_lev_max=0.1,
        learning_rate=1e-3, batch_size=32, test_batch_size=16,
        use_amp=False, seed=1,
        # energy score 只影響訓練，不影響推論；但 build_args 需要這些欄位
        lambda_cross_max=0.0, cross_timestep_ratio=1.0,
        cross_min_t_ratio=0.0, cross_gen_aligned=False, cross_gen_t_ratio=0.9,
        aux_no_teacher_forcing=False,
        lambda_energy_max=0.0, energy_n_samples=3,
        energy_min_t_ratio=0.6, energy_beta=1.0,
    )


def load_cycle(tag, k, out_dir, parameterization="x_start"):
    """載回第 k 期的模型與 scaler。"""
    cli = default_cli(tag); cli.parameterization = parameterization
    args = WF.build_args(cli)
    # Exp_Main 建構時會需要一個可讀的資料路徑（只用來初始化，不會真的訓練）
    args.root_path = os.path.join(out_dir, "windows") + os.sep
    args.data_path = f"win_{k:02d}.csv"
    args.wf_train_len, args.wf_val_len = WF.TRAIN_LEN, 0

    ck = os.path.join(out_dir, "ckpt", f"cycle_{k:02d}.pth")
    sc = os.path.join(out_dir, "ckpt", f"cycle_{k:02d}_scaler.pkl")
    for p in (ck, sc):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    exp = Exp_Main(args)
    exp.model.load_state_dict(torch.load(ck, map_location=DEVICE))
    exp.model.eval()
    with open(sc, "rb") as f:
        scaler = pickle.load(f)
    return exp, scaler, args


def get_conditions(scaler, k, out_dir, n_cond, rng):
    """從該期的訓練視窗取 n_cond 個互不重疊的 63 日 condition（已標準化）。"""
    win = pd.read_csv(os.path.join(out_dir, "windows", f"win_{k:02d}.csv"))
    cols = [c for c in win.columns if c != "date"]
    arr = scaler.transform(win[cols].values.astype(np.float32))
    T = arr.shape[0]
    starts = sorted(rng.choice(np.arange(0, T - SEQ_LEN), size=n_cond, replace=False))
    return [arr[s:s + SEQ_LEN] for s in starts], arr


@torch.no_grad()
def sample_same_condition(exp, cond_scaled, K):
    """同一個 condition 抽 K 次（每次 fresh noise）。回傳 (K, PRED_LEN, NUM_VARS)。"""
    cond_b = np.repeat(cond_scaled[np.newaxis], K, axis=0)
    x_enc = torch.tensor(cond_b, dtype=torch.float32, device=DEVICE)
    dec = np.concatenate([cond_scaled[-LABEL_LEN:],
                          np.zeros((PRED_LEN, NUM_VARS), dtype=np.float32)], axis=0)
    x_dec = torch.tensor(np.repeat(dec[np.newaxis], K, axis=0),
                         dtype=torch.float32, device=DEVICE)
    xme = torch.zeros(K, SEQ_LEN, 4, device=DEVICE)
    xmd = torch.zeros(K, LABEL_LEN + PRED_LEN, 4, device=DEVICE)
    out, *_ = exp.model.forward(x_enc, xme, x_dec, xmd, sample_times=1)
    out = out[:, 0] if out.dim() == 4 else out
    return out.cpu().numpy()


def condition_collapse(exp, conds, K, seed):
    """**逐 condition** 的變異數拆解：noise-driven vs cond-driven。

    對單一 condition 抽 K 個樣本 s(k, t, i)，令 mu(t,i) = mean_k s：
      var_noise = Var_k(s)              擴散隨機性造成的變異
      var_mean  = Var_{t,i}(mu)         condition 決定的確定性結構
      noise_frac = var_noise / (var_noise + var_mean)

    兩個實作上的要點：
      1) 必須**逐 condition** 算再取中位數。取平均會被少數「爆掉的取樣」主導 ——
         實測同一期的四個 condition 中，會有一個的 var_noise 是其他的 20 倍以上。
         那是取樣不穩定，不是模型有健康的隨機性。
      2) seed_corr 必須**扣掉條件均值**再算。原始 corr(sample0, sample1) 會被
         「不同 ticker/時點的均值本來就不同」灌水到 0.96，看起來像完全忽略噪聲，
         但那是均值的跨元素變異造成的假象。
    """
    torch.manual_seed(seed)
    rows = []
    for j, c in enumerate(conds):
        s = sample_same_condition(exp, c, K)[:, :, RCLOSE_IDX]   # (K, 21, 10)
        mu = s.mean(axis=0)
        var_noise = float(((s - mu) ** 2).mean())
        var_mean = float(mu.var())
        resid = (s - mu).reshape(K, -1)
        rc = np.corrcoef(resid[0], resid[1])[0, 1] if resid[0].std() > 0 else np.nan
        raw = np.corrcoef(s[0].ravel(), s[1].ravel())[0, 1]
        rows.append(dict(cond=j, var_noise=var_noise, var_mean=var_mean,
                         noise_frac=var_noise / (var_noise + var_mean + 1e-12),
                         seed_corr_raw=float(raw), seed_corr_resid=float(rc),
                         sample_std=float(s.std())))
    df = pd.DataFrame(rows)
    # 爆掉的取樣：var_noise 超過中位數 10 倍
    med = df.var_noise.median()
    df["unstable"] = df.var_noise > 10 * med
    return df


def build_cond_ts(model, cond_scaled):
    """複製 DDPM.forward 的前處理，回傳 cond_ts (1, 30, 63+21=84)。

    與 exp_denoiser_sensitivity.py:build_cond_ts 相同，維持兩份診斷可比。
    """
    x_enc = torch.tensor(cond_scaled[np.newaxis], dtype=torch.float32, device=DEVICE)
    lin = model.dlinear_model(
        x_enc.permute(0, 2, 1)[:, :, -model.short_term_range:]).permute(0, 2, 1)   # (1,21,30)
    x_past = torch.cat([x_enc.permute(0, 2, 1), lin.permute(0, 2, 1)], dim=-1)     # (1,30,84)
    return x_past, lin


@torch.no_grad()
def denoiser_sensitivity(exp, cond_scaled, M, t_list):
    """固定 cond，餵 M 個不同的 x0 訊號，看 denoiser 的輸出是否追蹤 x_t。

    slope ~ 1、ratio ~ 1  → denoiser 有把 x_t 的資訊帶到輸出（正常）
    slope ~ 0、ratio << 1 → contraction map：輸出幾乎與 x_t 無關（收縮）
    """
    model = exp.model
    dw = model.diffusion_worker
    dw.eval()
    cond_ts, _ = build_cond_ts(model, cond_scaled)
    rows = []
    for t in t_list:
        x0m = torch.randn(M, NUM_VARS, PRED_LEN, device=DEVICE)
        tt = torch.full((M,), t, device=DEVICE, dtype=torch.long)
        x_t = dw.q_sample(x_start=x0m, t=tt, noise=torch.randn_like(x0m))
        raw = dw.nn(x_t, tt, cond_ts.expand(M, -1, -1).contiguous(), y_clean=None)
        # nn 的輸出依參數化而不同（x0 / eps / v），一律先換算回 x0 才能跨參數化比較
        if dw.parameterization == "noise":
            out = dw.predict_start_from_noise(x_t, tt, raw)
        elif dw.parameterization == "v":
            out = dw.predict_start_from_v(x_t, tt, raw)
        else:
            out = raw
        a = x0m.permute(0, 2, 1).cpu().numpy()[:, :, RCLOSE_IDX].reshape(M, -1)
        b = out.permute(0, 2, 1).cpu().numpy()[:, :, RCLOSE_IDX].reshape(M, -1)
        ac, bc = a - a.mean(0), b - b.mean(0)
        num = (ac * bc).mean(0)
        corr = float(np.nanmean(num / (ac.std(0) * bc.std(0) + 1e-12)))
        slope = float(np.nanmean(num / (ac.var(0) + 1e-12)))
        rows.append(dict(t=t, corr=corr, slope=slope,
                         ratio=float(b.std(0).mean() / (a.std(0).mean() + 1e-12))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dow10_wf_v2")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--cycles", type=int, nargs="+", default=[0, 13, 27])
    ap.add_argument("--n_cond", type=int, default=8)
    ap.add_argument("--K", type=int, default=32, help="同一 cond 的取樣次數")
    ap.add_argument("--M", type=int, default=64, help="denoiser 探針的樣本數")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parameterization", default="x_start",
                    choices=["x_start", "noise", "v"])
    ap.add_argument("--save", default="./outputs/wf_collapse_probe")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join("./outputs", a.tag)
    os.makedirs(a.save, exist_ok=True)
    rng = np.random.RandomState(a.seed)

    print("=" * 78)
    print(f"  Walk-Forward Collapse Probe   tag={a.tag}  cycles={a.cycles}  device={DEVICE}")
    print("=" * 78)

    cc_rows, ds_frames = [], []
    for k in a.cycles:
        print(f"\n── Cycle {k:02d} " + "─" * 60)
        exp, scaler, _ = load_cycle(a.tag, k, out_dir, a.parameterization)
        meta_p = os.path.join(out_dir, "cycles", f"cycle_{k:02d}_meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        print(f"   origin={meta.get('origin_date','?')}  best_step={meta.get('best_step','?')}")

        conds, _ = get_conditions(scaler, k, out_dir, a.n_cond, rng)

        cc = condition_collapse(exp, conds, a.K, a.seed + k)
        cc.insert(0, "cycle", k)
        cc.insert(1, "origin", meta.get("origin_date"))
        cc_rows.append(cc)
        nf = cc.noise_frac.median()
        verdict = ("COLLAPSE（輸出幾乎是 cond 的確定性函數）" if nf < 0.3
                   else "部分回應噪聲" if nf < 0.6 else "正常")
        print(f"   [condition collapse] noise_frac(中位數)={nf:.4f}  "
              f"seed_corr(去均值)={cc.seed_corr_resid.median():.4f}  "
              f"不穩定取樣 {int(cc.unstable.sum())}/{len(cc)}  → {verdict}")

        ds = denoiser_sensitivity(exp, conds[0], a.M, [1, 5, 10, 20, 40, 60, 80, 95])
        ds.insert(0, "cycle", k)
        ds_frames.append(ds)
        lo = ds[ds.t <= 10]
        print(f"   [denoiser contraction] 低 t 平均 slope={lo.slope.mean():.4f}  "
              f"out_std/x0_std={lo.ratio.mean():.4f}")

        del exp
        torch.cuda.empty_cache()

    cc_df = pd.concat(cc_rows, ignore_index=True)
    ds_df = pd.concat(ds_frames, ignore_index=True)
    cc_df.to_csv(os.path.join(a.save, "condition_collapse.csv"), index=False)
    ds_df.to_csv(os.path.join(a.save, "denoiser_sensitivity.csv"), index=False)

    print("\n" + "=" * 78)
    print("  彙總")
    print("=" * 78)
    print(cc_df.round(5).to_string(index=False))
    print()
    print("逐 cycle 中位數：")
    print(cc_df.groupby("cycle")[["var_noise", "var_mean", "noise_frac",
                                  "seed_corr_raw", "seed_corr_resid",
                                  "sample_std"]].median().round(5).to_string())
    print()
    print(ds_df.groupby("t")[["corr", "slope", "ratio"]].mean().round(4).to_string())
    print(f"\n已存：{a.save}/")

    nf, sc = cc_df.noise_frac.median(), cc_df.seed_corr_resid.median()
    print("\n【判讀】")
    if nf < 0.3:
        print(f"  noise_frac={nf:.3f} < 0.3、seed_corr={sc:.3f}")
        print("  → condition collapse 仍然存在。模型實質上是確定性回歸器，")
        print("    條件變異數 ~= 0，因此任何 stylized-fact loss 或把統計量加進 condition")
        print("    都無法改變邊際分布形狀。應先處理『輸出不隨 seed 變動』這件事")
        print("    （換獎勵分散度的目標函數，例如 energy score / CRPS）。")
    else:
        print(f"  noise_frac={nf:.3f}、seed_corr={sc:.3f}")
        print("  → 相較舊協定（noise_frac~0.001、seed_corr~0.999）已改善，前提改變，")
        print("    stylized-fact loss 值得重新嘗試。")


if __name__ == "__main__":
    main()
