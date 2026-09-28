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
6. F(H) 與 AR 分支保留。主訓練仍使用 future mixup，驗證及生成皆不提供真實未來。

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
  --out outputs/spx/initial

python3 spx_experiment.py sample \
  --checkpoint outputs/spx/initial/model.pt \
  --history-end 2023-12-29 \
  --condition -0.05 0.15 --condition 0.05 0.15 \
  --condition -0.05 0.35 --condition 0.05 0.35 \
  --n-paths 200 --out outputs/spx/initial_samples
```

`--condition -0.05 0.35` 表示未來 21 個交易日累積報酬 -5%、年化波動率 35%。
可重複 `--condition`。各條件使用相同初始亂數流以協助比較，輸入數值仍須有歷史聯合分布支持。
`--history-end` 必須是 CSV 內的交易日，且有至少 63 列歷史特徵。
`--data` 可指定具 `date,Close,High,Low` 欄位的價格 CSV；不會 forward-fill 缺失價格。
`--device cpu` 可指定 CPU。`--threads` 是全域選項，放在子命令之前。

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

## 第一輪實跑結果

下載了 6,289 筆 SPX 日價格。上述範例已完成 AR 500 步及 diffusion 2,000 步訓練，
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
