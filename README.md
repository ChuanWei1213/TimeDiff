# TimeDiff — Dow 10 Synthetic Financial Time Series

新增 SPX 單一指數實驗：固定預訓 AR，以未來 21 天的累積報酬與波動率控制生成。
操作與驗證方式見 [SPX_EXPERIMENT.md](SPX_EXPERIMENT.md)，入口為 `spx_experiment.py`。
SPX 預設停用 future mixup；分布對照見 [實驗 notebook](notebooks/spx_no_mixup_comparison.ipynb)。

所有產物的分類與現況見 [outputs 索引](outputs/README.md)；機器可讀版本為
[catalog.yaml](outputs/catalog.yaml)。Dow 10 依 `walk_forward`、`conditional` 與
`single_shot` 分開，PR30／PR10 批次共用模型的關係也記錄在索引中。

以 conditional diffusion 生成 Dow 10 個股的多變量日報酬序列，並以一套 τ 校準的
stylized-facts 框架評估生成品質。

**三件事看這份文件就夠：怎麼生資料、哪裡可以調、評估怎麼看。**

> **這是一份交接用的精簡包。** 程式碼完整，可重跑整條管線；產出資料只保留
> 評估與論文表格所需的部分。詳見文末 [§7 交接包內容](#7-交接包內容)。

```
資料前處理  →  訓練 + 生成  →  評估
preprocess     walk_forward     evaluation.run_all
               _dow10.py                ↓
                    ↓              results/
        outputs/dow10/walk_forward/timediff/<tag>/paths/
```

---

## 0. 環境

```bash
conda activate Marcro2          # torch 2.9.1+cu128, GPU
```

> 預設的 `/opt/anaconda3/bin/python` **無法訓練**（缺 `einops`，且 torch 為 CPU build）。
> 它可以跑 `evaluation/`（純 numpy/pandas/scipy），但 `walk_forward_dow10.py` 必須用 Marcro2。

---

## 1. 怎麼生資料

### 1.1 前處理（只需跑一次，資料已在 repo 內）

```bash
python data_provider/preprocess_walkforward_dow10.py
```

產出 `datasets/prediction/data_finance/`：

| 檔案 | 內容 |
|---|---|
| `dow10_wf_returns.csv` | 2012-01-04 ~ 2025-01-08，`date` + 10×{rclose, rhigh, rlow} |
| `dow10_wf_ohlc.csv` | 同期間的 Close/High/Low（價格重建的 seed 來源） |

定義：`rclose = log(C_t/C_{t-1})`、`rhigh = log(H_t/C_t)`、`rlow = log(L_t/C_t)`。

### 1.2 訓練 + 生成（同一支腳本）

```bash
python walk_forward_dow10.py --tag my_model \
  --max_steps 600 --eval_every 50 --val_batches 4 \
  --n_paths 50 --gen_seed 1 --save_ckpt --long_format \
  --parameterization v
```

約 20–30 分鐘（28 期 × 600 步）。產出 `outputs/dow10/walk_forward/timediff/my_model/`：

| 路徑 | 內容 |
|---|---|
| `paths/returns/path_XX_returns.csv` | **評估唯一會讀的東西**。`date` + 10×{rclose,rhigh,rlow}，1,761 列 |
| `paths/price/path_XXX.csv` | OHLC 長格式 `date,tic,close,high,low`（給下游 RL／回測） |
| `cycles/` | 每期生成結果 + meta（`--resume` 用） |
| `windows/` | 每期的訓練視窗 CSV（可再生） |
| `ckpt/` | 每期模型權重、scaler（`--generate_only` 用） |

### 1.3 只重新取樣，不重訓

已有 `ckpt/` 時：

```bash
python walk_forward_dow10.py --tag my_model --generate_only \
  --n_paths 500 --gen_seed 2025 --long_format --parameterization v
```

> ⚠️ 這會**覆蓋** `outputs/dow10/walk_forward/timediff/my_model/paths/`。若要保留原有路徑，先建新目錄並把 `ckpt`
> 做成 symlink，再用 `--out_dir` 指過去（`RL/path/` 的 500 條就是這樣生的）。

### 1.4 匯入外部模型的輸出

第三方模型（nsdiff / cndiff / armd / dva…）的資料格式各異，用轉檔器統一：

預設寫入 `outputs/dow10/walk_forward/external/<tag>/paths/returns/`。
既有統計模型 tag 會維持在 `statistical/` 分類。

```bash
# 先 dry-run 確認格式判斷正確
python -m evaluation.prepare_model_output --tag nsdiff \
  --input <你的檔案.csv> --kind price|return --layout long|wide --dry-run

# 沒問題再拿掉 --dry-run 寫檔
```

| 參數 | 說明 |
|---|---|
| `--kind` | `price`（會自動轉成 log return，必要時以真實前一日 Close 當 seed）或 `return` |
| `--layout` | `long`（`date,tic,value`）或 `wide`（`date` + `{TICKER}_xxx`） |
| `--value-col` | 數值欄名稱會誤導時明確指定（例：return 值放在名為 `close` 的欄位） |
| `--path-col` | 單檔內含多路徑時，標示路徑編號的欄位 |

**硬性要求**：日期必須與真實交易日**逐日完全一致**（primary window 為 1,761 天）。
對不上會直接 raise 並列出缺哪幾天 —— 這是刻意的，不容許靜默對齊。

---

## 2. 哪裡可以調

### 2.1 模型與生成（`walk_forward_dow10.py` 的 CLI）

**最常用**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--parameterization` | `x_start` | `x_start` / `noise` / `v`。**建議用 `v`** |
| `--max_steps` | — | 每期訓練步數上限 |
| `--n_paths` | 1 | 生成幾條路徑 |
| `--gen_seed` | 0 | 生成用的隨機種子（與訓練種子分離） |
| `--save_ckpt` | off | 存模型權重，`--generate_only` 需要 |
| `--long_format` | off | 價格輸出為 `date,tic,close,high,low` |

**結構化修正**（各自獨立，可疊加；全部預設關閉）

| 參數 | 作用 |
|---|---|
| `--whiten_cross_asset` | Cholesky 白化：把跨資產相關以線性變換裝上去 |
| `--vol_scaling` | GARCH(1,1) 遞迴時變波動：波動聚集成為結構性保證 |
| `--heavy_tail_residual` | Student-t 邊際校準（保序，保留 copula） |
| `--vol_common` | σ 加入市場共同成分（**實測有害**，僅供 ablation） |

**實驗性 loss**（預設關閉，實測效果有限）

| 參數 | 作用 |
|---|---|
| `--lambda_cross_max` | Cross-asset correlation loss 權重 |
| `--aux_no_teacher_forcing` | 輔助 loss 改用 `y_clean=None` 的額外 forward |
| `--cross_gen_aligned` | Cross loss 改用純噪聲輸入 + 高噪聲 t |
| `--lambda_energy_max` | Energy score |

**控制實驗**

| 參數 | 作用 |
|---|---|
| `--force_step N` | 跳過 Stage A 選模，直接訓練 N 步（排除訓練長度的干擾） |
| `--max_cycles N` | 只跑前 N 期（除錯用） |
| `--resume` | 沿用既有的 `cycles/*.npy` |

### 2.2 實驗協定（`walk_forward_dow10.py` 頂部的全域常數）

```python
SEQ_LEN   = 63     # condition 長度
LABEL_LEN = 21
PRED_LEN  = 21     # 每個生成區塊的長度
TRAIN_LEN = 1260   # 每期訓練窗
STRIDE    = 63     # 重訓間隔
```

> ⚠️ 改 `PRED_LEN` 必須同步改 `evaluation/config.py` 的 `WALKFORWARD_PRED_LEN`。
> 那裡有一行 `assert ACF_MAX_LAG < WALKFORWARD_PRED_LEN` 在擋 —— 因為 lag ≥ PRED_LEN 的
> 時間相依性有一部分來自真實 condition 而非模型，納入評估會系統性高估模型。

### 2.3 評估設定（`evaluation/config.py`）

| 設定 | 預設 | 說明 |
|---|---|---|
| `MODELS_WHITELIST` | — | **要評估哪些模型**，填模型 tag；搬移後 tag 不變 |
| `MODEL_DISPLAY_NAMES` | — | 表格中顯示的名稱 |
| `PRIMARY_EVAL_START/END` | 2018-01-02 / 2024-12-31 | 主表排名唯一依據 |
| `ROBUSTNESS_WINDOWS` | 2020 Crisis, 2022 Bear | 只作 regime 分析，不參與排名 |
| `BOOTSTRAP_B` | 500 | τ 校準的重抽次數 |
| `SF_WEIGHTING` | `hierarchical` | `hierarchical`（distribution 0.5 / dynamics 0.5）或 `equal` |
| `W_SF` / `W_CA` | 0.5 / 0.5 | Overall 的兩軸權重 |
| `ACF_MAX_LAG` | 20 | VC / LE 曲線的 lag 上限 |
| `JUMP_Q` | 0.95 | Jump 門檻的機率水平（由真實資料定義） |

**加入新模型只要兩步**：資料放進 `outputs/dow10/walk_forward/<分類>/<tag>/paths/returns/` → 把 `<tag>` 加進
`MODELS_WHITELIST`。τ 與模型無關，**加入新模型不會改變既有模型的分數**。

分類為 `timediff`、`statistical`、`external`；評估工具會掃描三類。
單次生成則放在 `outputs/dow10/single_shot/<tag>/`。

### 2.4 修改模型本體

| 想改什麼 | 去哪裡 |
|---|---|
| Loss 的組合 | `models_diffusion/diffusion_worker.py` `forward()`（主 loss 與各輔助 loss 相加處） |
| Stylized loss 的實作 | `models_diffusion/stylized_loss.py` |
| Condition 怎麼進網路 | `models_diffusion/DDPM.py` `train_forward()` |
| Denoiser 架構 | `models_diffusion/DDPM_modules/DDPM_CNNNet.py` |
| 推論時的 condition 組裝 | `walk_forward_dow10.py` `generate_block()` |

---

## 3. 評估怎麼看

### 3.1 執行

```bash
python -m evaluation.run_all          # 約 2–3 分鐘
```

或逐節閱讀版本：`evaluation_framework.ipynb`（20 個 section，結果與上面完全一致）。

### 3.2 核心概念：分數是怎麼來的

所有指標都先算一個**距離** $D$，再轉成 $[0,1]$ 的分數：

$$\text{Score} = 2^{-D/\tau}$$

$\tau$ 是「真實資料自己跟自己比」的誤差水準（stationary block bootstrap，**只用真實資料**，
因此不存在用被評估對象決定評估標準的循環問題）。

| $D/\tau$ | Score | 解讀 |
|---|---|---|
| 0 | 1.000 | 與真實完全一致 |
| **1** | **0.500** | **誤差恰等於真實資料自身的抽樣誤差** |
| 2 | 0.250 | 2 倍 |
| 4 | 0.063 | 4 倍 |

所以看到分數 0.06，直接讀成「**誤差是真實市場自然隨機性的 4 倍**」。

### 3.3 產出檔案

```
results/
├── tables/
│   ├── table_SF1_scores.csv          五個 stylized facts 的分數（越高越好）
│   ├── table_SF2_distances.csv       同樣五項的 raw distance（越低越好）
│   ├── table_SF3_dispersion.csv      逐檔離散度：平均是否被少數股票拉高
│   ├── table_crossasset_scores.csv   Corr / PCA / Tail Dep. 分數
│   ├── table_crossasset_distances.csv 對應的 raw distance
│   ├── table_overall_comparison.csv  主表：Method × Time
│   ├── table_signal_to_noise.csv     各指標的統計功效
│   ├── table_real_metric_CI.csv      真實 metric 的信賴區間
│   └── asset_level_<tag>.csv         逐檔明細（論文 appendix 用）
├── figures/           7 類圖，檔名用 model tag
├── figure_data/       每張圖的完整數值（txt）
├── evaluation_methodology_report.txt   17 節方法論
└── paper_results_summary.txt           12 節結果摘要
```

### 3.4 讀表的五個要點

**① raw distance 與 score 分開兩張表，不要混看。** SF-2 是原始誤差（可檢核），
SF-1 是經 τ 無量綱化後的分數（可跨指標比較）。

**② 只看 primary window 排名。** 主表有 `Role` 欄：`PRIMARY (ranking)` 才決定排名，
2020 / 2022 是 regime robustness，**不可與 primary 混在一起比較**。

**③ 注意 `Low-power metrics` 欄。** 訊噪比 $|\theta_{\text{real}}|/\tau < 1$ 的指標，
真實訊號本身就淹沒在估計噪音裡，該欄的分數**鑑別力不足**。Negative Skewness 與
Leverage Effect 在 primary window 就接近臨界。

**④ 年度表與全期表不可直接比較。** τ 逐窗重新校準，T≈252 的年度窗 τ 較大（容忍度寬），
分數普遍高於全期。年度之間可比，跨尺度不可比。

**⑤ VolRatio 不在分數內。** 它是尺度而非 stylized fact，刻意排除，但在
`table_crossasset_distances.csv` 獨立報告。一個 VolRatio 0.2 的模型，
相關結構可能很好但波動只有真實的五分之一 —— 對下游用途是致命的，分數卻看不出來。

### 3.5 逐年報表（另一套）

```bash
python run_yearly_report.py       # 2018–2024 每年一張表 + 綜合表
python make_return_plots.py       # AAPL 報酬分布圖
```

產出 `results/yearly_report/`。表格把 SF-1 與 Cross-Asset 合併成單一張，
所有欄位都是 10 檔平均的 normalized score。

圖有三張，各回答不同問題：

| 圖 | 看什麼 |
|---|---|
| `aapl_return_density.png` | **振幅** —— 線性縱軸的鐘形圖，峰越高代表波動越小 |
| `aapl_return_shape.png` | **形狀** —— 標準化後與 N(0,1) 對照，排除振幅差異 |
| `aapl_return_tails.png` | **尾部** —— 對數縱軸，藍線斷掉處即該方法產生過的最極端事件 |

### 3.6 模型診斷工具

分數只告訴你「好不好」，不告訴你「為什麼」。三支診斷腳本回答後者：

```bash
python exp_wf_collapse_probe.py --tag <tag> --parameterization v --cycles 0 13 27
```

| 輸出 | 意義 |
|---|---|
| `noise_frac` | 輸出變異中有多少來自擴散噪聲。**< 0.3 代表模型退化成確定性回歸器** |
| 低 t 斜率 | denoiser 對 $x_t$ 的響應。應接近 $\bar\alpha \approx 1$；遠低於此代表 $x_t$ 路徑失效 |
| `sample_std` | 標準化空間的樣本標準差，目標 1.0 |

`--cycles` 可指定要探查哪幾期（0 = 第一期，27 = 最後一期）。

---

## 4. 目前的模型

下表為既有模型清單；目前檔案位於 `outputs/dow10/walk_forward/` 的各分類中。
2026-10-18 搬移檢查發現部分舊 CSV 原本已有長度／日期不一致，未修改資料；
可讀性與期間覆蓋請以 [outputs 索引](outputs/README.md) 為準。

| tag | 說明 |
|---|---|
| `dow10_wf_v2` | 原始 baseline（x_start parameterization） |
| `dow10_wf_vpred` | **v-parameterization**，單層修正 |
| `v_whiten` | v + Cholesky 白化 |
| `v_3layer` | v + 白化 + GARCH 波動縮放 |
| `v_4layer_fix` | v + 白化 + 波動縮放 + 重尾校準（四層完整版） |
| `v_cross` / `v_cross_ga` | Cross-asset loss 的兩個變體 |
| `nsdiff` / `cndiff` / `armd` / `dva` | 外部對照模型 |

`RL/path/` 另有 500 條價格路徑（`dow10_wf_vpred`，`gen_seed=2025`，與評估用的 50 條
獨立，避免下游 RL 與評估資料重疊）。

---

## 5. 文件索引

| 文件 | 內容 |
|---|---|
| `PIPELINE.md` | 完整 pipeline、相依鏈、可刪檔案清單 |
| `results/RESEARCH_PROGRESS_REPORT.md` | 研究進度報告（診斷鏈、四層結構、結果） |
| `results/PROBLEM_SETTING.md` | 問題設定（中英文） |
| `results/TABLE_SF1_methodology.md` | 五個 stylized facts 的 distance 設計與理由 |
| `results/TABLE_CrossAsset_methodology.md` | 三個 cross-asset 指標的設計與理由 |
| `evaluation_framework_design.md` | 評估框架的設計決策紀錄 |
| `results/evaluation_methodology_report.txt` | 自動產生的 17 節方法論（隨每次評估更新） |

---

## 6. 已知限制

1. **Walk-forward 不是 free-running simulation**：每 21 個交易日重新錨定到一段真實的
   63 日 condition。因此 lag ≥ 21 的時間相依性有一部分來自真實資料，**本評估不能推論
   模型的長期自主模擬能力**。這也是 `ACF_MAX_LAG = 20` 的直接原因。

2. **結構化修正是先驗而非學習**：白化的 $\Sigma$、GARCH 參數、Student-t 自由度都由訓練窗
   估計後施加，不是模型學到的。論文須明確區分「結構施加」與「模型學習」的貢獻。

3. **訓練與生成的固有落差**：訓練時 $x_t$ 由真實資料加噪而得，生成時從純噪聲出發。
   因此**訓練期量測到的統計量不能代表生成行為**（實測：低 t 時訓練期跨資產相關 0.481，
   純噪聲輸入僅 −0.001）。任何評估都必須在完整生成鏈上進行。

4. **單一 seed**：所有完整訓練均為 `gen_seed=1` 的單次執行，未做多 seed 重複。

---

## 7. 交接包內容

這份 repo 是從完整實驗環境（5.7 GB）精簡而來（1.6 GB），刪掉的都是**可再生**或
**已被排除的舊實驗**。程式碼與文件是完整的。

### 有什麼

| 項目 | 內容 |
|---|---|
| 程式碼 | 前處理、訓練/生成、評估框架、診斷工具 — 全部齊全 |
| 資料 | `datasets/.../dow10_wf_returns.csv`、`dow10_wf_ohlc.csv`（前處理產物，不需重跑） |
| 生成結果 | 14 個模型的 `paths/returns/` — **足以重現所有評估表格** |
| Checkpoint | 只有 `dow10_wf_vpred/ckpt/`（參考模型，332 MB） |
| 價格路徑 | 只有 `dow10_wf_vpred/paths/price/`（格式範例）；`RL/path/` 500 條（下游用） |
| 報告 | `results/` 全部 — 表格、圖、方法論文件、研究進度報告 |

### 沒有什麼（以及怎麼拿回來）

| 缺少 | 為什麼 | 怎麼再生 |
|---|---|---|
| 其他 13 個模型的 `ckpt/` | 每個 332 MB，只有 `--generate_only` 需要 | 用 §1.2 的指令重訓（單模型約 20–30 分鐘） |
| 各模型的 `windows/`、`cycles/` | 訓練時自動產生的中間檔 | 重訓時自動重建 |
| 其他模型的 `paths/price/` | 由 `paths/returns/` + OHLC 決定性重建 | 重跑生成，或用 `reconstruct_ohlc()` |
| 舊 sweep 實驗（60+ 個 `outputs/` 目錄） | 結論已寫入 `research_experiment_report.md`，配置均已被排除 | 不建議重跑 |

### 接手第一步

```bash
conda activate Marcro2
python -m evaluation.run_all          # 約 3–5 分鐘，重現 results/tables/ 全部主表
```

跑得出來就代表環境正確。接著讀 `results/RESEARCH_PROGRESS_REPORT.md` 了解已做過什麼、
哪些路走不通，再從 §2.4 開始改模型。
