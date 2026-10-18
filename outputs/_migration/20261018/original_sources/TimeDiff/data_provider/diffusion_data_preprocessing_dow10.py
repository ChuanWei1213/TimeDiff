"""
diffusion_data_preprocessing_dow10.py
======================================
為 DOW Jones 10 支股票進行資料前處理，輸出 dow10_returns.csv。

特徵計算方式與原始 financial_returns.csv 相同：
  rclose = log(adj_close_t / adj_close_{t-1})
  rhigh  = log(high_t  / adj_close_t)
  rlow   = log(low_t   / adj_close_t)

訓練期間：2010-01-01 ~ 2023-12-31
輸出特徵數：30 (10 stocks × 3 features)
欄位順序：AAPL_rclose, AAPL_rhigh, AAPL_rlow, AMGN_rclose, ...
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
START_DATE = "2010-01-01"
END_DATE   = "2023-12-31"

OUTPUT_DIR  = "./datasets/prediction/data_finance"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "dow10_returns.csv")
PRICE_CSV   = os.path.join(OUTPUT_DIR, "dow10_adj_close.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"開始下載 DOW10 資料 ({START_DATE} ~ {END_DATE})...")
print(f"Tickers: {TICKERS}")

# ==========================================
# 下載資料
# ==========================================
raw = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=True)

adj_close = raw["Close"][TICKERS]
adj_high  = raw["High"][TICKERS]
adj_low   = raw["Low"][TICKERS]

print(f"原始資料筆數: {len(adj_close)}")

# ==========================================
# 計算 log returns
# ==========================================
r_close = np.log(adj_close / adj_close.shift(1))           # cross-day return
r_high  = np.log(adj_high  / adj_close)                    # intraday up
r_low   = np.log(adj_low   / adj_close)                    # intraday down

# ==========================================
# 組合成 30 維特徵，按照 ticker 排列
# ==========================================
cols_order = []
for t in TICKERS:
    cols_order += [f"{t}_rclose", f"{t}_rhigh", f"{t}_rlow"]

feat_df = pd.concat(
    {
        "rclose": r_close.rename(columns=lambda c: c),
        "rhigh":  r_high.rename(columns=lambda c: c),
        "rlow":   r_low.rename(columns=lambda c: c),
    },
    axis=1
)
# 重新命名欄位
feat_df.columns = [f"{t}_{feat}" for feat, t in feat_df.columns]
feat_df = feat_df[cols_order].dropna()

print(f"清理後筆數（dropna）: {len(feat_df)}")
print(f"日期範圍: {feat_df.index[0].date()} ~ {feat_df.index[-1].date()}")

# ==========================================
# 存檔
# ==========================================
# TimeDiff data_loader.py 要求日期欄命名為小寫 'date'
# yfinance 回傳的 DatetimeIndex.name 是 'Date'（大寫），需手動修正
feat_df.index.name = "date"   # 改為小寫
feat_df_out = feat_df.reset_index()  # 把 date 從 index 移到欄位
feat_df_out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Log-return 特徵已儲存：{OUTPUT_CSV}")
print(f"   Shape: {feat_df_out.shape}  (rows × cols, 30 features + 1 date)")

# 同時儲存初始 adj_close（用於生成時反推絕對股價）
adj_close.dropna().to_csv(PRICE_CSV)
print(f"✅ Adj Close 已儲存：{PRICE_CSV}")

# ==========================================
# 簡單統計確認
# ==========================================
print("\n=== 簡單統計（rclose 部分）===")
rclose_cols = [f"{t}_rclose" for t in TICKERS]
print(feat_df[rclose_cols].describe().round(5))
