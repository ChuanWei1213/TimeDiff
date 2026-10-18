# TimeDiff — Pipeline 與相依鏈

## 1. Pipeline 全貌

```
[1] 前處理（資料準備，只需跑一次）
    python data_provider/preprocess_walkforward_dow10.py
        ↓
    datasets/prediction/data_finance/
        dow10_wf_returns.csv     2012-01-04 ~ 2025-01-08，date + 10×{rclose,rhigh,rlow}
        dow10_wf_ohlc.csv        同期間的 Close/High/Low（價格重建的 seed 來源）

[2] 訓練 + 生成（同一支腳本包辦）
    python walk_forward_dow10.py --tag <tag> --max_steps N --n_paths P --save_ckpt
        ↓
    outputs/<tag>/
        paths/returns/path_XX_returns.csv    ← evaluation 唯一會讀的東西
        paths/price/path_XXX.csv             OHLC 長格式（給下游 RL/回測用）
        cycles/                              每期生成結果 + meta（--resume 用）
        windows/                             每期的訓練視窗 CSV（可再生）
        ckpt/                                每期模型權重（--generate_only 用）

[3] 外部模型轉檔（nsdiff / cndiff / armd / dva）
    python -m evaluation.prepare_model_output --tag <tag> --input <csv> \
           --kind price|return --layout long|wide [--value-col X] [--dry-run]
        ↓
    outputs/<tag>/paths/returns/path_XX_returns.csv   （轉成同一格式）

[4] 評估
    # 先把 tag 加進 evaluation/config.py 的 MODELS_WHITELIST
    python -m evaluation.run_all
        ↓
    results/tables/*.csv            SF-1 / SF-2 / SF-3 / cross-asset / overall / SNR
    results/figures/*.png           7 類圖
    results/figure_data/*.txt       每張圖的完整數值
    results/evaluation_methodology_report.txt
    results/paper_results_summary.txt

[5] 論文報告（可選，內容與 [4] 相同，只是逐節呈現）
    evaluation_framework.ipynb
```

**你的理解正確**：`walk_forward_dow10.py` 訓練 → `python -m evaluation.run_all` 出指標。
中間唯一的橋樑是 `outputs/<tag>/paths/returns/*.csv`。

## 2. walk_forward_dow10.py 的相依鏈

```
walk_forward_dow10.py
├── exp/exp_main.py                          訓練迴圈
│   └── models_diffusion/DDPM.py             模型主體（condition 的組成處）
│       ├── DDPM_modules/DDPM_CNNNet.py      denoiser（CNN UNet）
│       ├── diffusion_worker.py              擴散過程 + loss 組合
│       │   └── stylized_loss.py             五個 stylized facts 的可微 loss
│       └── samplers/dpm_sampler.py          DPM-Solver（寫死 CUDA）
├── data_provider/data_factory.py            被 monkey-patch 換成 DatasetWF
│   └── data_loader.py  (Dataset_Custom)
├── utils/timefeatures.py
└── generate_dow10.py                        ← 只借用兩個函式，不可刪
        compute_stylized_facts, print_and_save_stylized_facts
```

## 3. 接下來要改 loss / condition 的確切位置

### Loss

| 檔案 | 位置 | 內容 |
|---|---|---|
| `models_diffusion/stylized_loss.py` | `volatility_clustering_loss` (108)、`leverage_effect_loss` (143)、`fat_tail_loss` (268)、`gpd_tail_loss` (370) | 五個 stylized loss 的可微實作 |
| | `compute_stylized_loss` (448) | 把上面幾個組起來、套用 annealing |
| | `anneal_linear` (393) / `anneal_sigmoid` (413) | 權重排程 |
| `models_diffusion/diffusion_worker.py` | 204–230 | **主 loss 與 stylized loss 相加的地方**（`loss_simple` + `loss_vlb` + stylized） |
| `walk_forward_dow10.py` | `build_args` 的 Stylized Facts Loss 區塊（約 272–290） | `use_volatility_loss`、`lambda_vol_max`、`anneal_ratio` 等開關 |

已存在但預設關閉的 loss：`use_fat_tail_loss`、`use_gpd_tail_loss`（先前實驗證實對 Hill 無效，
但程式碼完整可用）。

### Condition

| 檔案 | 位置 | 內容 |
|---|---|---|
| `models_diffusion/DDPM.py` | `train_forward` (83–121) | `x_enc` → `x_past`，含 window normalization 的開關 |
| | `pretrain_forward` (55–) | DLinear 預測分支 |
| `models_diffusion/DDPM_modules/DDPM_CNNNet.py` | — | condition 如何餵進 denoiser（`cond_ddpm_num_layers`、`cond_ddpm_channels_conv`） |
| `walk_forward_dow10.py` | `generate_block` (453–480) | **推論時的 condition 組裝**（`x_enc` / `x_dec` / `x_mark`） |
| | 全域常數 (70–72) | `SEQ_LEN=63`、`LABEL_LEN=21`、`PRED_LEN=21` |

> ⚠️ **改 `PRED_LEN` 會連動 evaluation**：`evaluation/config.py` 有
> `assert ACF_MAX_LAG < WALKFORWARD_PRED_LEN`。若把生成區塊拉長／縮短，
> 必須同步更新 `WALKFORWARD_PRED_LEN`，否則 VC / LE 的 lag 上限會失去依據。

### 診斷工具（改 condition 時直接可用）

`exp_wf_collapse_probe.py` — walk-forward 協定下的統一診斷工具，一次給出三件事：

- **condition collapse**：固定 condition、只換擴散噪聲，量測 `noise_frac`
  （輸出變異中來自噪聲的比例；< 0.3 代表模型退化成確定性回歸器）
- **denoiser 敏感度**：把輸出換算回 $\hat{x}_0$ 後對 $x_0$ 回歸，斜率應接近 $\bar\alpha$
- **sample_std**：標準化空間的樣本標準差，目標 1.0

```bash
python exp_wf_collapse_probe.py --tag <tag> --parameterization v --cycles 0 13 27
```

這支正是先前診斷出「denoiser 是 contraction map、condition collapse」的工具，
你要改 condition 時會直接用到。

### 改完之後的評估流程

```bash
python walk_forward_dow10.py --tag timediff_v3_newloss --max_steps 600 --n_paths 50 --save_ckpt
# 把 "timediff_v3_newloss" 加進 evaluation/config.py 的 MODELS_WHITELIST 與 MODEL_DISPLAY_NAMES
python -m evaluation.run_all
```

新模型會自動與 baseline / nsdiff / cndiff / armd 並列在同一張主表，
τ 不會因為加入新模型而改變，所以**舊分數不需要重算、可直接比較**。

---
