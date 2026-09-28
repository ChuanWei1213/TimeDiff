# SPX 條件式 TimeDiff 實驗

本入口只訓練及生成 S&P 500 指數（Yahoo 代號 `^GSPC`）。一個標的、三個特徵：
`rclose = log(C_t/C_{t-1})`、`rhigh = log(H_t/C_t)`、`rlow = log(L_t/C_t)`。
保留 63 天歷史 → 21 天未來。既有 Dow 10 腳本與輸出可繼續使用。

## 模型與訓練設定

1. AR 使用最近 21 天，先以未來 21 天的 MSE 預訓。
2. 預訓完成後設定 `requires_grad_(False)`，從 optimizer 排除 AR；結束時檢查權重未變。
3. 每個 target 的 21 個原始尺度收盤對數報酬計算兩個條件：
   - 累積簡單報酬：`expm1(sum(rclose))`，包含歷史最後收盤到未來第一天的報酬。
   - 年化波動率：`std(rclose, ddof=1) * sqrt(252)`。
4. 條件 scaler 僅以訓練 target 擬合；特徵 scaler 僅以訓練期間資料擬合，兩者隨 checkpoint 保存。
5. 條件經 MLP 產生 FiLM 的 scale/shift，注入每一步去噪 CNN。
6. F(H) 與 AR 分支保留。SPX 預設移除 future mixup：真實未來只用於加噪與 loss target，
   不混入去噪網路的條件。訓練加上 `--future-mixup` 可重現原本的對照設定。
   舊 checkpoint／其他 runner 未設定此旗標時維持原本行為。

預設 `v`、100 級噪聲、20 步 DPM-Solver、64 個隱藏通道；CPU/CUDA 皆可。
為隔離市場條件的作用，本入口不啟用 stylized-facts 輔助損失、GARCH、白化或振幅校正。
這些設定與既有 Dow 10 runner 不完全相同，應使用相同 SPX 入口做消融對照。

## 執行

在 `TimeDiff/` 執行。需要 Python、torch、numpy、pandas、tqdm、matplotlib 與 yfinance。
這個入口不需要 einops。下載會保存價格、日期、設定、套件版本及 SHA-256；同一路徑已存在時不覆寫。

```bash
python3 spx_experiment.py download --start 2000-01-01 --end 2025-01-01

python3 spx_experiment.py train \
  --train-end 2021-12-31 --val-end 2023-12-31 \
  --ar-steps 500 --steps 2000 \
  --out outputs/spx/no_mixup

python3 spx_experiment.py sample \
  --checkpoint outputs/spx/no_mixup/model.pt \
  --history-end 2023-12-29 \
  --condition -0.05 0.15 --condition 0.05 0.15 \
  --condition -0.05 0.35 --condition 0.05 0.35 \
  --n-paths 200 --out outputs/spx/no_mixup_samples
```

`--condition -0.05 0.35` 表示未來 21 個交易日累積報酬 -5%、年化波動率 35%。
可重複 `--condition`。各條件使用相同初始亂數流以協助比較，輸入數值仍須有歷史聯合分布支持。
`--history-end` 必須是 CSV 內的交易日，且有至少 63 列歷史特徵。
`--data` 可指定具 `date,Close,High,Low` 欄位的價格 CSV；不會 forward-fill 缺失價格。
`--device cpu` 可指定 CPU。`--threads` 是全域選項，放在子命令之前。
AR 預訓、diffusion 訓練、驗證及取樣皆顯示 tqdm 進度；`--no-progress` 可關閉。
訓練進度顯示當前 loss 與最佳驗證分數，每次驗證後即寫入 `training.csv`。
metadata 保存完整訓練 CLI、程式 commit、套件版本及 mixup 開關。

只有整段 target 都位於同一切分區間才會使用；驗證／測試可回看切分前的歷史，
但 target 不跨區間。範例將 2024 年留作測試，訓練與選模不使用它。
樣本窗口會重疊，不能把窗口數當成獨立樣本數。

## 輸出與解讀

訓練輸出 `model.pt`、`metadata.json`、`training.csv`、`train_conditions.csv`。
預設每 100 步以驗證集前 4 個 batch 做完整取樣，按條件標準化後的均方誤差選 checkpoint。
這是開發用的條件符合度指標；不代表完整驗證期的市場真實性，也不是事前預測績效。
增大 `--val-batches` 可覆蓋更多驗證窗口。最終測試期不參與選模。

生成輸出：

| 檔案 | 內容 |
|---|---|
| `paths.csv` | 每條 21 天路徑，含原始及修正後 high/low 特徵、價格、目標條件 |
| `path_metrics.csv` | 每條路徑自己的報酬、波動率與條件命中結果 |
| `summary.csv` | 平均、MAE、5/95 分位數、聯合命中率、符號修正比例及最近訓練條件距離 |
| `condition_distributions.png` | 生成統計量的分布與目標虛線；不是平均價格路徑的波動率 |
| `sampling.json` | 歷史截止日、種子、條件、樣本數與容許誤差 |
| `history_features.csv` | 生成實際使用的 63 天歷史特徵 |

命中率預設允許報酬絕對誤差 0.02（2 個百分點）、年化波動率誤差 0.05。
可用 `--return-tolerance` / `--vol-tolerance` 調整；不是相對百分比。
最近訓練條件距離在標準化的二維空間計算，只是外插診斷，不是生成有效性的保證。
價格以歷史最後收盤延伸。`rhigh` 截至不小於 0、`rlow` 截至不大於 0；raw 欄位保留修正前結果。
未來日期以 `step=1..21` 表示，避免擅自假設交易所假日。
這些是互相獨立的情境片段，不會串接成長期合成市場。

## 對照與驗證

相同設定加上 `--no-market-condition`，並使用另一個 `--out`，可訓練固定 AR、無市場條件的對照。
即使 baseline 的 sample 指定不同 condition，它也不接收它們；固定 seed 下輸出應相同。
兩組應使用相同資料切分、步數與樣本數。完整研究還需比較多個歷史窗口、隨機種子、
波動聚集、尾端／回撤及下游 RL 表現，不能僅靠條件命中率判定成功。

```bash
python3 -m pytest -p no:cacheprovider tests/test_spx.py -q
```

測試涵蓋標記定義、切分、凍結 AR、條件梯度、三種參數化、兩種 sampler、CPU、
奇數 batch、舊版無條件 API、推論不讀真實未來，以及 checkpoint 到報表的完整流程。

為讓本入口可測試，亦修正共用 sampler 的 CPU 裝置選擇、非 100 步時間索引、
DDPM 的 v 還原，以及訓練奇數 batch 的噪聲步數配對。

## 第一輪實跑結果（原始 mixup 對照）

下載了 6,289 筆 SPX 日價格。先前啟用 mixup 的版本已完成 AR 500 步及 diffusion 2,000 步訓練，
依驗證條件誤差選中第 100 步 checkpoint；AR 權重經檢查未變。
固定 2023-12-29 截止的歷史，每個條件取樣 200 條：

| 目標報酬 | 目標年化波動率 | 生成平均報酬 | 生成平均年化波動率 | 聯合命中率 |
|---:|---:|---:|---:|---:|
| -5% | 15% | -1.17% | 14.81% | 25.5% |
| +5% | 15% | +6.03% | 14.31% | 25.5% |
| -5% | 35% | +4.32% | 16.58% | 0% |
| +5% | 35% | +11.19% | 16.06% | 0% |

命中標準為報酬 ±2 個百分點、波動率 ±5 個百分點，兩者同時滿足。
此版本已跑通條件傳遞，但尚未達成可靠的高波動率控制。
四組的 high 特徵截斷比例約 22%–31%，low 約 8%–25%，亦需改善；
不能因修正後價格合法就認為生成品質合格。
這只是單一歷史窗口、單一種子的開發診斷，尚未完成多窗口測試或 RL 評估。
詳細數值及分布圖位於 `outputs/spx/initial_samples/`。

## 第 1 階段：移除 mixup

本階段只改變 future mixup，尚未新增條件一致性 loss 或 CFG。使用前述 `no_mixup`
指令，保持原始資料、切分、模型大小、學習率、訓練步數、seed、取樣條件與樣本數。
每組依同一驗證規則選擇 checkpoint，最佳步數可能不同。

原始對照已保存在 `outputs/spx/initial`；若在全新 checkout 重跑對照，使用：

```bash
python3 spx_experiment.py train --future-mixup \
  --train-end 2021-12-31 --val-end 2023-12-31 \
  --ar-steps 500 --steps 2000 --out outputs/spx/initial
python3 spx_experiment.py sample \
  --checkpoint outputs/spx/initial/model.pt --history-end 2023-12-29 \
  --condition -0.05 0.15 --condition 0.05 0.15 \
  --condition -0.05 0.35 --condition 0.05 0.35 \
  --n-paths 200 --out outputs/spx/initial_samples
```

既有輸出資料夾不會被覆寫，重跑時請指定新路徑並更新 notebook 的 `RUNS`。

開啟 [spx_no_mixup_comparison.ipynb](notebooks/spx_no_mixup_comparison.ipynb)，
由上而下執行可重算並比較：

- 每條路徑的報酬／波動率分布與目標容許區間。
- 二維條件分布、MAE、聯合命中率、OHLC 符號修正比例。
- 訓練與驗證曲線、最佳 checkpoint、固定編號的路徑樣貌。

Notebook 會檢查兩組資料指紋、歷史、已記錄的設定與樣本數，從 `paths.csv` 重算並
核對已保存指標。圖表與比較 CSV 匯出至 `outputs/spx/no_mixup_comparison/`。
Notebook 保留執行結果於 Git；大型資料與 checkpoint 維持本機保存。

條件指定的是兩個統計量，不是完整機率分布。沒有假設生成路徑應符合常態分布；
分位數是樣本分布範圍，不是信賴區間。移除 mixup 改變亂數消耗，同 seed 不代表
所有訓練噪聲與 batch 順序完全配對。本輪只是一個歷史窗口、一個訓練 seed 的診斷。

### 本輪結果

程式版本 `2005511` 完成 AR 500 步、diffusion 2,000 步，選中第 900 步 checkpoint。
最佳驗證條件 MSE 為 0.485747（原始 mixup：1.049102）。兩組預訓 AR 權重逐項相同，
本輪訓練後 AR 權重亦保持不變。以下各組皆為 200 條完整生成路徑：

| 目標報酬 | 目標年化波動率 | 無 mixup 平均報酬 | 無 mixup 平均波動率 | 原始命中率 | 無 mixup 命中率 |
|---:|---:|---:|---:|---:|---:|
| -5% | 15% | -4.34% | 14.89% | 25.5% | 42.5% |
| +5% | 15% | +2.64% | 14.23% | 25.5% | 28.5% |
| -5% | 35% | -4.74% | 28.76% | 0% | 14.5% |
| +5% | 35% | +4.33% | 26.25% | 0% | 5.5% |

四組等權平均：報酬 MAE 由 6.41 降至 3.71 個百分點，波動率 MAE 由 10.38 降至
4.66 個百分點，聯合命中率由 12.75% 提升至 22.75%。四組的兩種 MAE 都降低，
但 35% 波動率仍明顯偏低；+5%／15% 這組的平均報酬雖更偏離目標，逐路徑 MAE 仍下降，
顯示平均值與分布誤差必須一起看。不能將這輪改善直接推論為 RL 表現提升。

原始結果：`outputs/spx/no_mixup_samples/summary.csv`；notebook 會從 `paths.csv` 獨立重算核對。

本環境的 Jupyter TCP 核心無法啟動，因此本輪以 IPython 同程序模式由上而下執行
全部 8 格 Python 程式，保存表格及 4 張圖，指標核對全部通過，並人工檢查匯出圖表。
一般 Jupyter 環境仍可直接 Run All；若同樣受限於本機連接埠，可使用：

```bash
python3 notebooks/execute_in_process.py notebooks/spx_no_mixup_comparison.ipynb
```

此輔助程式只支援本實驗的一般 Python／Matplotlib 輸出，不等同完整 Jupyter 核心或 widget 驗證。

## 暴跌條件重跑（七組）

在 app 終端執行以下指令，可以直接觀察 tqdm：

```bash
bash scripts/run_spx_crash.sh
```

腳本會切換至專案根目錄，依序：

1. 以相同資料切分、seed 2024、AR 500 步與 diffusion 2,000 步重新訓練無 mixup 模型。
2. 對原有四組條件，以及報酬／年化波動率 `(-10%, 40%)`、`(-20%, 60%)`、`(-30%, 80%)` 各取樣 200 條。
3. 沿用既有 `initial/model.pt` mixup 對照，重新生成相同七組條件。
4. 執行 [暴跌條件 notebook](notebooks/spx_crash_comparison.ipynb)，核對路徑指標、繪圖並檢查訓練資料支持。

新增條件只用於推論，不會改變訓練資料或標記。本輪仍未加入條件一致性 loss 或 CFG。
報酬與波動率同時改變，不能將結果歸因於其中一個條件。訓練窗口的局部支持亦不代表當前歷史背景下合理。

所有新資料與完整進度日誌位於 `outputs/spx/crash_20260929/`；重複啟動同名實驗會中止，避免覆寫。
第二次重跑可用 `bash scripts/run_spx_crash.sh crash_repeat2`。Notebook 由環境變數 `SPX_CRASH_RUN`
選取資料目錄；手動開啟時也可修改第一個程式格的 `RUN_DIR`。
Python 預設為終端的 `python3`，可用 `SPX_PYTHON=/absolute/path/to/python3` 指定已安裝依賴的直譯器。
進度列會即時顯示，並同時寫入該輪 `run.log`。腳本不會覆寫前一輪實驗資料，會更新暴跌 notebook 的執行輸出。

本節為已準備的重跑流程；實際結果以完成後的 notebook 與該輪 `summary.csv` 為準。

## 第 2 階段：條件一致性 loss（lambda = 0.1）

在無 mixup 的模型上新增 `--lambda-condition 0.1`：

`L_total = L_diffusion + lambda * (L_return + L_volatility)`。

模型輸出先依 v／noise／x_start 參數化還原為 `x0_hat`，反標準化第 0 個特徵，
逐路徑計算 `expm1(sum(rclose))` 與 `std(rclose, ddof=1)*sqrt(252)`。
兩項誤差分別除以**訓練標記**的標準差後平方，最後跨 batch 平均；兩個 MSE 相加，沒有再除以 2。
lambda=0.1 等同報酬與波動率兩項各乘 0.1。整個計算保留梯度，不做路徑後處理。

預設 lambda=0，保持舊行為。啟用時要求市場條件存在、停用 mixup 與窗口正規化。
使用同一次 forward 的乾淨路徑估計，不增加噪聲抽樣，也不新增隨機分支。
時間點抽樣沿用原本設定，不額外做時點加權或權重暖身。
這是加噪真實樣本的**單步還原** loss，不保證完整生成符合條件；仍用原本完整取樣的驗證規則選模。

```bash
python3 scripts/run_spx_consistency.py
```

腳本執行同設定 AR 500／diffusion 2,000 步與七組各 200 條取樣，
沿用 `crash_20260929` 的無 mixup、lambda=0 結果作對照；不重訓對照、不加入 CFG。
結果位於 `outputs/spx/condition_lambda_0p1/`，重跑須指定新的 `--tag`，避免覆寫。
[比較 notebook](notebooks/spx_condition_loss_comparison.ipynb) 會核對設定與原始路徑，
比較分布、逐路徑指標、loss 分量及計時。

計時檔案：

- `train/timings.json`：資料／模型準備、AR 預訓、diffusion 更新、驗證、其他記錄及 checkpoint I/O；子階段互不重疊。
- `samples/timings.json`：七組個別生成秒數、總生成時間及準備／指標／繪圖／匯出時間。
- `pipeline_timings.json`：訓練、取樣、notebook 各程序的實際 wall-clock 秒數（含 import／啟動），以及全流程耗時。

`training.csv` 分別記錄 diffusion loss、兩項原始條件 MSE 與加權條件懲罰，避免將不同總 loss 直接比較。
計時是本次機器的實測值，不以它比較舊版本速度；對照資料沿用，因此本輪對照不另計重訓時間。
