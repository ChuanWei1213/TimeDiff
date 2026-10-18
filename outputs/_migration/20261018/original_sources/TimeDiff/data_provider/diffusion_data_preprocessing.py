"""
diffusion_data_preprocessing.py
================================
下載 5 檔股票的 10 年日線資料 (auto_adjust=True 除權息還原)，
計算三種 Log Return 特徵：
  r_close : 跨天 Close 對數收益率 (動能)
  r_high  : High / Close  對數比  (恆 >= 0)
  r_low   : Low  / Close  對數比  (恆 <= 0)

輸出 CSV 格式相容於 TimeDiff Dataset_Custom：
  - 第 0 欄為 'date'
  - 欄位順序：date, NVDA_rclose, NVDA_rhigh, NVDA_rlow,
                    AAPL_rclose, AAPL_rhigh, AAPL_rlow, ...
  - 最後一欄固定命名為 'OT' (TimeDiff 的 target 預設欄位)

另外也儲存：
  - adj_close_clean.csv  (初始價格，用於推論期反推絕對價格)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 設定參數
# ==========================================
TICKERS    = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
START_DATE = "2015-01-01"
END_DATE   = "2023-12-31"

OUTPUT_DIR = "./datasets/prediction/data_finance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"開始下載並進行除權息/拆股還原數據 ({START_DATE} ~ {END_DATE})...")

# ==========================================
# 下載數據
# ==========================================
df_adj = yf.download(
    TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True
)

adj_close = df_adj["Close"][TICKERS]
adj_high  = df_adj["High"][TICKERS]
adj_low   = df_adj["Low"][TICKERS]

print("計算 Log Returns 特徵...")

# ==========================================
# 計算 3 大 Log Return 特徵
# ==========================================
# (A) Close 的跨天對數收益率 (動能)
r_close = np.log(adj_close / adj_close.shift(1))

# (B) High 相對於當天 Close 的對數比  (恆 >= 0)
r_high = np.log(adj_high / adj_close)

# (C) Low 相對於當天 Close 的對數比   (恆 <= 0)
r_low = np.log(adj_low / adj_close)

# 清除第一天因為 shift(1) 產生的 NaN，並對齊所有 Index
r_close = r_close.dropna()
r_high  = r_high.loc[r_close.index]
r_low   = r_low.loc[r_close.index]
adj_close_clean = adj_close.loc[r_close.index]

# ==========================================
# 組合成 TimeDiff Dataset_Custom 相容的 CSV
# 欄位順序：date, T0_rclose, T0_rhigh, T0_rlow, T1_rclose, ...
# ==========================================
frames = []
for ticker in TICKERS:
    frames.append(r_close[ticker].rename(f"{ticker}_rclose"))
    frames.append(r_high[ticker].rename(f"{ticker}_rhigh"))
    frames.append(r_low[ticker].rename(f"{ticker}_rlow"))

combined = pd.concat(frames, axis=1)

# TimeDiff Dataset_Custom 需要：
#   1) 日期欄叫 'date'
#   2) 最後一欄叫 'OT'（target，重新命名最後一欄）
combined.index.name = "date"
combined = combined.reset_index()

# 把最後一欄 rename 成 'OT' 讓 TimeDiff 的 --target OT 可以正常使用
last_col = combined.columns[-1]
combined = combined.rename(columns={last_col: "OT"})

# ==========================================
# 存檔
# ==========================================
out_path       = os.path.join(OUTPUT_DIR, "financial_returns.csv")
adj_close_path = os.path.join(OUTPUT_DIR, "adj_close.csv")

combined.to_csv(out_path, index=False)
adj_close_clean.to_csv(adj_close_path)   # 保留 date index，用於反推價格

print(f"\n✅ 特徵矩陣已儲存：{out_path}")
print(f"   形狀 (含 date 欄): {combined.shape}")
print(f"   欄位: {list(combined.columns)}")
print(f"\n✅ 原始調整後 Close 已儲存：{adj_close_path}")
print(f"   形狀: {adj_close_clean.shape}")
print(f"\n共 {len(combined)} 個交易日，{len(TICKERS)} 檔股票，每檔 3 特徵 => num_vars = {len(TICKERS)*3}")