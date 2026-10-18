# SPX 2020–2025 市場狀態搜尋

資料：Yahoo Finance 的 S&P 500 指數 `^GSPC`，未調整日線 OHLC。
來源：https://finance.yahoo.com/quote/%5EGSPC/history/

- 原始資料：2018-12-03 至 2025-12-31，共 1,779 根。
- 目標期間：2020-01-02 至 2025-12-31，共 1,508 根；之前 271 根作為暖機。
- 與 XNYS 日曆核對，沒有缺漏／額外交易日、重複日期或無效 OHLC。

## 固定規則

1. 每根 K 棒掃描 21–63 根的搜尋窗口，尋找最高價、最低價及各自位置；重複極值取最近一次。
2. 事件長度 `k = abs(high_idx-low_idx)+1`，本身也必須介於 21–63 根。端點包含在內。
3. 在辨識當下回看 252 根，包含當根。對每個 k 保存 `252-k+1` 個完整滾動區段的 TR。
4. 區段 TR = `max(區段最高價, 區段前收盤) - min(區段最低價, 區段前收盤)`。
5. 門檻 = 相同 k 的 TR 算術平均 + 1 個母體標準差（ddof=0）。所有數值以 SPX 指數點表示。
6. 高低價差嚴格超過門檻，且低點先出現 → 上漲；高點先出現 → 下跌。相等或同根則為調整。
7. 同方向、相同起訖的重複辨識去重；部分重疊、端點不同的區段仍保留。
8. 圖上以合格事件的日期聯集塗色。藍色：僅上漲；橘色：僅下跌；紫色：漲跌重疊；灰色：調整（未被此尺度的合格事件覆蓋）。

這是回溯事件標記，不是事件起點當時即可取得的交易訊號。首次辨識日另存於事件表；計算不使用 2025 年後資料。連續色帶可能由多個重疊事件構成，因此可長於 63 根；690 個不同區段不代表 690 次獨立市場事件。

## 結果

- 上漲：511 個不同起訖區段；下跌：179 個不同起訖區段。
- 日期互斥分類：上漲 574 天、下跌 224 天、重疊 9 天、調整 701 天。
- 28 項單元測試通過。
- 690 個事件的平均與標準差另外以 pandas 滾動切片交叉驗證，最大絕對誤差約 4.18e-12。

## 檔案

- `spx_market_states_overview.png` / `.svg`：六年總覽。
- `spx_market_states_by_year.png` / `.svg`：逐年放大，縱軸各自獨立。
- `trend_events.csv`：不同起訖事件及首次合格辨識的門檻、日期、延遲。
- `trend_detections.csv`：所有合格辨識紀錄（同一天同區段只留一筆）。
- `daily_market_states.csv`：每日 OHLC、漲跌事件覆蓋數與狀態。
- `yearly_state_counts.csv`：年度互斥分類交易日數。
- `spx_daily_ohlc.csv`：包含暖機期的資料快照。
- `summary.json`：參數、資料指紋、來源、摘要與驗證資訊。
- `calculation_validation.json`：獨立重算驗證結果。

## 重現

於專案根目錄執行：

```bash
MPLCONFIGDIR=/private/tmp/finrl-market-states-mpl venv/bin/python -m finrl.experiments.run_market_state_scan
venv/bin/python -m pytest unit_tests/experiments/test_market_state_tracker.py -q
```

重現使用已保存的 CSV，不需重新連線下載；相關套件為 pandas、numpy、matplotlib、exchange_calendars。
