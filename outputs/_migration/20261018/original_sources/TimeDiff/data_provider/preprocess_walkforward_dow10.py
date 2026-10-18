"""
preprocess_walkforward_dow10.py
================================
Walk-Forward 實驗專用前處理（全新獨立檔案，不影響 diffusion_data_preprocessing_dow10.py）。

與原前處理的差異：
  1. 期間拉長到 2012-01-01 ~ 2025-01-10
     （第一個 origin = 2018-01-02，需往前 1,260 個共同交易日 → 約 2013-01）
  2. 額外輸出 OHLC 原始價格檔（Close/High/Low），供
     - 價格重建的唯一 seed（2017 最後一個共同交易日的真實 Close）
     - 與真實股價比較繪圖
  3. dropna() 後的索引即為 10 檔股票的「共同交易日」

特徵定義與訓練時完全一致：
  rclose = log(C_t / C_{t-1})
  rhigh  = log(H_t / C_t)
  rlow   = log(L_t / C_t)

輸出：
  ./datasets/prediction/data_finance/dow10_wf_returns.csv   (date + 30 features)
  ./datasets/prediction/data_finance/dow10_wf_ohlc.csv      (date + 30 price cols)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 設定參數
# ==========================================
TICKERS    = ["AAPL", "AMGN", "CRM", "CSCO", "IBM",
              "INTC", "MSFT", "NKE", "VZ", "WMT"]

# 需求：預測期 2018-01-02 ~ 2024-12-31，第一個 origin 前需 1,260 個共同交易日。
# 1,260 交易日 ≈ 5 年 → 從 2012-01-01 下載可留足夠緩衝。
START_DATE = "2012-01-01"
END_DATE   = "2025-01-10"   # yfinance 的 end 為 exclusive，多抓幾天確保含 2024-12-31

OUTPUT_DIR   = "./datasets/prediction/data_finance"
RETURNS_CSV  = os.path.join(OUTPUT_DIR, "dow10_wf_returns.csv")
OHLC_CSV     = os.path.join(OUTPUT_DIR, "dow10_wf_ohlc.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"開始下載 DOW10 資料 ({START_DATE} ~ {END_DATE})...")
print(f"Tickers: {TICKERS}")

raw = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=True)

adj_close = raw["Close"][TICKERS]
adj_high  = raw["High"][TICKERS]
adj_low   = raw["Low"][TICKERS]

print(f"原始資料筆數: {len(adj_close)}")

# ==========================================
# 共同交易日：三個價格表都非 NaN 的日期
# ==========================================
price = pd.concat({"Close": adj_close, "High": adj_high, "Low": adj_low}, axis=1)
price = price.dropna()
print(f"共同交易日筆數（Close/High/Low 皆完整）: {len(price)}")

adj_close = price["Close"]
adj_high  = price["High"]
adj_low   = price["Low"]

# ==========================================
# 計算 log returns（與訓練時一致）
# ==========================================
r_close = np.log(adj_close / adj_close.shift(1))   # cross-day return
r_high  = np.log(adj_high  / adj_close)            # intraday up
r_low   = np.log(adj_low   / adj_close)            # intraday down

cols_order = []
for t in TICKERS:
    cols_order += [f"{t}_rclose", f"{t}_rhigh", f"{t}_rlow"]

feat_df = pd.concat({"rclose": r_close, "rhigh": r_high, "rlow": r_low}, axis=1)
feat_df.columns = [f"{t}_{feat}" for feat, t in feat_df.columns]
feat_df = feat_df[cols_order].dropna()   # 第一列因 shift(1) 為 NaN，被移除

print(f"清理後筆數（dropna）: {len(feat_df)}")
print(f"日期範圍: {feat_df.index[0].date()} ~ {feat_df.index[-1].date()}")

# ==========================================
# 檢查：預測期天數與 walk-forward 排程
# ==========================================
pred_mask = (feat_df.index >= "2018-01-02") & (feat_df.index <= "2024-12-31")
n_pred    = int(pred_mask.sum())
n_hist    = int((feat_df.index < "2018-01-02").sum())
print(f"\n預測期 2018-01-02 ~ 2024-12-31 共同交易日: {n_pred} 天 (規格為 1761)")
print(f"第一個 origin 之前可用歷史天數: {n_hist} 天 (需 >= 1260)")
if n_pred != 1761:
    print(f"⚠️  實際天數 {n_pred} != 1761；walk_forward_dow10.py 會依實際天數推導期數與最後一期長度。")
if n_hist < 1260:
    raise ValueError(f"歷史天數不足：{n_hist} < 1260，請把 START_DATE 往前調。")

# ==========================================
# 存檔
# ==========================================
feat_df.index.name = "date"
feat_df.reset_index().to_csv(RETURNS_CSV, index=False)
print(f"\n✅ Log-return 特徵已儲存：{RETURNS_CSV}  shape={feat_df.shape}")

ohlc_cols = {}
for t in TICKERS:
    ohlc_cols[f"{t}_Close"] = adj_close[t]
    ohlc_cols[f"{t}_High"]  = adj_high[t]
    ohlc_cols[f"{t}_Low"]   = adj_low[t]
ohlc_df = pd.DataFrame(ohlc_cols)
ohlc_df = ohlc_df.loc[feat_df.index]        # 對齊 return 的日期索引
ohlc_df.index.name = "date"
ohlc_df.reset_index().to_csv(OHLC_CSV, index=False)
print(f"✅ OHLC 價格已儲存：{OHLC_CSV}  shape={ohlc_df.shape}")

seed_date = feat_df.index[feat_df.index < "2018-01-02"][-1]
print(f"\n價格重建 seed（2017 最後一個共同交易日）= {seed_date.date()}")
print(ohlc_df.loc[seed_date, [f'{t}_Close' for t in TICKERS]].round(2).to_string())
