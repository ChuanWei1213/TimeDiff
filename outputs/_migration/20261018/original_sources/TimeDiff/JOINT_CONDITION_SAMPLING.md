# Dow 10 聯合條件取樣

`sample_joint_dow10.py` 從訓練期間的真實視窗抽取完整的 **歷史 H、21 日報酬 R、年化波動 sigma** 配對，交由既有條件式 TimeDiff 生成多條未來路徑。這個入口只做取樣、生成與診斷，不會啟動 PPO。

## 取樣定義

- 預設條件母體為 2010–2017，歷史 63 日、目標 21 日。目標的每個交易日必須完整位於訓練期內；歷史脈絡可以早於起始日。
- R 與 sigma 沿用 `data_provider/dow10_cond.py`：Dow 10 每日等權再平衡組合的 21 日累積報酬，以及日 log return 樣本標準差乘上 sqrt(252)。
- 以母體 R 的線性內插 PR30 作為尾部門檻，包含等於門檻的視窗。分位數、抽樣與配對不使用 2018 年之後的資料。
- 固定 70% 抽樣名額來自完整池，30% 來自尾部池；兩池內都等機率、有放回抽樣，最後打散順序。抽樣數不是 10 的倍數時，完整池名額採四捨五入，並記錄實際比例。
- 完整池也包含尾部，所以實際尾部條件的理論占比約為 `0.7 × 0.3 + 0.3 = 51%`，不是 30%。遇到相同報酬值時依實際尾部樣本數重算。
- 每次抽樣保留原本的 H 與其對應 R、sigma；不獨立抽報酬／波動，也不把條件任意配到另一個歷史起點。
- 每組條件預設生成 4 條路徑；推論 decoder 的未來部分為零。原始真實未來只用於計算條件及事後診斷。
- 不依生成結果剔除路徑、重抽條件或調整條件值。

## 執行

在 TimeDiff 根目錄執行。每個輸出路徑都必須尚不存在，連空目錄也會拒絕覆寫。

只建立可檢查的取樣計畫，不需要 checkpoint：

```bash
.venv-spx/bin/python sample_joint_dow10.py plan \
  --data datasets/dow10/prices_2004.csv \
  --train-start 2010-01-01 --train-end 2017-12-31 \
  --n-draws 500 --paths-per-condition 4 \
  --full-weight 0.7 --tail-quantile 0.3 --plan-seed 20261017 \
  --out outputs/joint_condition_plan_new
```

使用條件式 Dow 10 checkpoint 生成 2,000 條路徑並診斷：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/private/tmp/timediff-joint-mpl \
  .venv-spx/bin/python sample_joint_dow10.py generate \
  --data datasets/dow10/prices_2004.csv \
  --checkpoint outputs/dow10_joint_20261017/train/model.pt \
  --train-start 2010-01-01 --train-end 2017-12-31 \
  --n-draws 500 --paths-per-condition 4 \
  --full-weight 0.7 --tail-quantile 0.3 \
  --plan-seed 20261017 --diffusion-seed 20261018 \
  --batch-size 32 --threads 4 --device cpu \
  --out outputs/joint_condition_batch_new
```

生成器訓練期可以早於條件母體，但結束日必須相同；資料檔 SHA-256、資產順序、通道、63／21 日長度與兩個條件維度都會核對。沒有重新擬合 checkpoint 的 scaler，也沒有修改模型結構。

計畫種子與 diffusion 種子分開記錄。固定種子可以重現同一設定下的抽樣；生成結果的逐位元重現另依賴 PyTorch 版本、裝置與 batch size。

## 輸出

| 檔案 | 內容 |
|---|---|
| `generation.json` | 執行狀態、實際比例、種子、設定、套件版本、checkpoint／資料／程式／結果雜湊 |
| `population.csv` | 全部合格視窗、歷史／起點／目標日期、R／sigma、尾部標記、理論混合機率 |
| `sampling_plan.csv` | 每次抽樣的 draw_id、window_id、來源池與完整配對 |
| `paths.npz` | 股票 OHLC、raw features、目標／實際條件、起點與配對稽核陣列 |
| `path_metrics.csv` | 每條路徑的條件偏差、回撤、股票相關性、最大單日變動與平方報酬 lag-1 相關 |
| `diversity_by_condition.csv` | 同一歷史視窗的重複路徑數、兩兩距離與相關、對原始未來的距離 |
| `distribution_quantiles.csv` | 訓練母體、抽中條件與生成路徑的邊際分布摘要 |
| `joint_distribution_bins.csv` | 使用訓練分位數切格的報酬／波動聯合分布；包含空格 |
| `sampling_by_year.csv` | 各年份母體、預期機率與實際抽樣次數 |
| `diagnostics.json` | 條件誤差、尾部占比、分布差異、多樣性與品質限制 |
| `README.md`、兩張 PNG | 可閱讀的診斷摘要、分布圖及同條件多路徑圖 |

`paths.npz` 的主要欄位維持既有 `SyntheticPaths` 所需格式。**`scenario` 在此只有來源池的相容標籤：0＝完整池，1＝尾部池，不再代表固定跌幅。** 完整池也可能抽中下跌尾部。

`history_features` 與 `paired_real_future` 按 draw_id 儲存一次；每條生成路徑的 `draw_id` 可回查它們。`target`、`realized`、來源池與真實未來是生成／稽核標籤，不能直接放入 PPO observation。

## 多樣性解讀

檢查完全重複之外，也計算同條件路徑的兩兩標準化 RMSE、去除每條路徑各資產均值後的 RMSE，以及組合逐日報酬相關。標準化固定使用 checkpoint 的訓練特徵尺度。

固定 R、sigma 時，期末報酬與波動差異小是合理現象，不能單憑它判定塌縮。反過來，零完全重複也不能證明路徑真實或多樣性充分。圖中按目標報酬排序固定選擇六個視窗，不依生成結果挑圖；虛線為配對真實路徑。

這是訓練支持範圍內的診斷，不是樣本外泛化或 PPO 有效性的證據。重疊視窗並非獨立事件；21 日內的波動聚集指標也較不穩定。近似複製檢查只比對各自原始未來，未搜尋生成器全部訓練路徑。

## 測試

```bash
MPLCONFIGDIR=/private/tmp/timediff-joint-mpl .venv-spx/bin/python -m pytest \
  tests/test_joint_conditions.py tests/test_spx.py -q \
  -o cache_dir=/private/tmp/timediff-joint-pytest-cache
```

測試涵蓋時間邊界、完整配對、70／30 來源名額、尾部重疊、固定種子、日期／資料污染防護、未來 decoder 不洩漏、已知複製／同統計不同形狀路徑，以及小型模型到完整輸出與覆寫拒絕的流程。測試中的隨機初始化小模型僅用於程式驗證，不是本次正式生成資料。
