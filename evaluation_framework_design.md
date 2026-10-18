# Financial Fidelity Evaluation Framework — Design Document (Phase 1–3)

狀態：**設計提案，尚未實作**。待確認後才建立 `evaluation_framework.ipynb`。

---

## PHASE 1 — Existing Implementation Audit

### 1.1 元件盤點

| Component | Existing Code | Status | Reuse / Rewrite |
|---|---|---|---|
| Stylized facts 主函式 | `generate_dow10.py:334 compute_stylized_facts` | 可用但只有 lag-1 | **Rewrite**（保留介面） |
| 同上（重複實作） | `compare_ablation.py:64`, `generate_10yr.py:368`, `eval_p0p1.py:26 facts`, `diagnostic_analysis.py:103`, `STYTEST.py:25` | 5 份重複、彼此不一致 | **統一為單一 module** |
| Hill tail index | `generate_dow10.py:305`（左尾）、`eval_p0p1.py:19`（雙尾） | **兩份定義不同** | **Rewrite**（定左尾 + k-based） |
| Jump | `generate_dow10.py:317 jump_measure`（BNS RJ） | **退化，見 BUG-1** | **Replace**（保留為診斷欄） |
| Volatility clustering | `np.corrcoef(r[:-1]**2, r[1:]**2)`，僅 lag 1 | 不足 | **Rewrite** → ACF curve |
| Leverage effect | `np.corrcoef(r[:-1], np.abs(r[1:]))`，僅 lag 1、用 \|r\| 非 r² | 不足 | **Rewrite** → LE curve |
| Skewness | `scipy.stats.skew`（`bias=True`） | 可用 | **Reuse**（改 `bias=False`） |
| Cross-asset correlation | `eval_p0p1.py:37 corr_mae` | 正確（取上三角、排除對角） | **Reuse** |
| Lower tail dependence | `eval_p0p1.py:42 lower_tail_dep` | **違反 threshold 規範，見 BUG-4** | **Rewrite** |
| PCA / eigenvalue spectrum | 不存在 | — | **New** |
| Bootstrap / CI | 不存在 | — | **New** |
| Multi-lag ACF 工具 | `diagnostic_part1.py:258` | 可參考 | **Reuse 概念** |
| 資料載入 | `eval_p0p1.py` 於 import 時 `yf.download()` | **不可重現，見 BUG-7** | **Rewrite**（改讀本地 CSV） |
| 模型輸出掃描 | 不存在（`eval_p0p1.py` 硬寫 tag） | — | **New** |

### 1.2 Bug 清單

**BUG-1（severity: 高）BNS Relative Jump 在 synthetic 上完全退化**

- 位置：`generate_dow10.py:317`，`RJ = max(0, (RV−BV)/RV)`
- 現象：walk-forward path 0 的 10 檔股票，未裁切的 `(RV−BV)/RV` 全為負（−0.10 ~ −0.20），
  經 `max(0, ·)` 後**10/10 全部等於 0**。真實資料 2024 平均 RJ = 0.2495、2018–2024 = 0.0933。
- Impact：Jump 這一欄對所有 synthetic model 都會是 0 → **零鑑別力**，主表該欄無意義。
- 成因：denoiser 是 contraction map，生成序列過度平滑，bipower variation 反而大於 realized
  variance（無跳躍時 RV≈BV，樣本噪音使 RV−BV 隨機為負）。
- Proposed fix：主指標改為 §PHASE 3 的 threshold-based jump（frequency + magnitude）；
  RJ 保留為**未裁切**的診斷欄（報 `(RV−BV)/RV` 原值，不做 `max(0,·)`）。

**BUG-2（中）Hill estimator 兩份定義不一致**

- `generate_dow10.py:305` 只用負報酬 `r[r<0]`（左尾）；`eval_p0p1.py:19` 用 `|r|`（雙尾混合）。
- Impact：兩支腳本產出的 Hill 數值不可互相比較，論文引用時會自相矛盾。
- Fix：統一為**左尾**（下跌尾部才是金融風險關心的對象），並在報告中明寫定義。

**BUG-3（高）Hill 尾部樣本數過小**

- 現行 `q=0.95` 套在負報酬上：T=252 → 負報酬約 120 筆 → 尾部 k ≈ 6。
- 由 Hill 的漸近變異 `SE(α) ≈ α/√k`，k=6 時相對 SE ≈ **40%**。
- Impact：α 的差異幾乎全是估計噪音，Heavy-Tail score 不可信。
- Fix：改用 k-based 門檻 `k = ⌈n_neg^{2/3}⌉`（T=252 → k≈24；T=1761 → k≈93），
  並報 k ∈ {0.5k, k, 2k} 的敏感度。

**BUG-4（高）`lower_tail_dep` 讓 real 與 synthetic 各自取自己的分位數**

- `eval_p0p1.py:42`：`thr = np.quantile(X, q, axis=0)`，X 是被評估的序列本身。
- Impact：違反「threshold 必須由真實資料定義」。但**直接改成真實 threshold 會使指標崩潰**
  （見 PHASE 2 的量化證據：24/32 個模型在真實 q05 之下的曝險數 < 5）。
- Fix：見 §PHASE 3 的 Tail Dependence 設計（雙軌：copula 版為主、absolute 版為輔）。

**BUG-5（中）Leverage effect 用 |r| 而非 r²、且只有 lag 1**

- `np.corrcoef(r[:-1], np.abs(r[1:]))`。文獻標準為 `Corr(r_t, r²_{t+k})`（Bouchaud et al. 2001）。
- Fix：改為 r² 並擴為 k = 1…K 曲線。

**BUG-6（中）Volatility clustering 只有 lag 1**

- 單一 lag 無法代表 clustering 的**衰減形狀**（真實市場為緩慢冪律衰減）。
- Fix：ACF(r², k) 曲線，k = 1…K。

**BUG-7（中）`eval_p0p1.py` 於 import 時連網下載 yfinance**

- Impact：結果不可重現（資料隨下載時間變動）、離線無法執行。
- Fix：一律讀 `datasets/prediction/data_finance/dow10_wf_returns.csv`。

**BUG-8（低）Family A 輸出無 date 欄**

- `outputs/*/dow10_generated_returns_2024.csv` 共 30 欄、252 列，第一欄即 `AAPL_rclose`。
- Fix：preprocessing layer 以序位對映到真實 2024 的 252 個交易日，並 assert 長度相符。

> 以上皆**不修改生成端程式碼**，全部在 evaluation 層處理。

### 1.3 資料格式盤點

**真實資料（reference）**

| 檔案 | 範圍 | 列數 | 欄位 |
|---|---|---|---|
| `dow10_wf_returns.csv` | 2012-01-04 → 2025-01-08 | 3,274 | `date` + 10×{rclose,rhigh,rlow} |
| `dow10_wf_ohlc.csv` | 同上 | 3,274 | `date` + 10×{Close,High,Low} |
| `dow10_returns.csv` | 2010-01-05 → 2023-12-29 | 3,521 | 同格式（**不含 2024**） |

- 10 檔 ticker 缺值：**全部為 0**，資料完整。
- Evaluation 的真實基準採用 `dow10_wf_returns.csv`（唯一涵蓋 2018–2024 全段者）。

**Synthetic 輸出：三個互不相容的家族**

| Family | 檔案樣式 | 目錄數 | 日期範圍 | 有 date 欄 | 多路徑 |
|---|---|---|---|---|---|
| **A** 單次 2024 | `dow10_generated_returns_2024.csv` | **32** | 2024（252 日，隱含） | ✗ | ✗ |
| **B** walk-forward | `paths/returns/path_XX_returns.csv` | 2（`dow10_wf`, `dow10_wf_v2`） | 2018-01-02 → 2024-12-31（1,761 日） | ✓ | ✓（50 條） |
| **C** 10yr ablation | `generated_log_returns_10yr.csv` | 1 | — | ✗ | ✗ |

- **Family C 必須排除**：其 ticker 是 NVDA/AAPL/MSFT/GOOGL/AMZN，不是 Dow 10。
- Family A 與 B **沒有共同的評估期間以外的交集**：A 只有 2024，B 有 2018–2024。
  2024 是唯一的共同支撐。

**Walk-forward 對齊（§25 檢查結果）**

| 檢查項 | 結果 |
|---|---|
| condition 只用過去真實資料 | ✓ block j 的 cond = `real[origin+21j−63 : origin+21j]` |
| 使用未來資料 | ✓ 無 |
| 價格串接方式 | ✓ `reconstruct_ohlc` 由 2017 最後一日的真實 Close 單一 seed 全段累積，區塊與重訓邊界都不重置 |
| evaluation 只用生成期 | ✓ 2018-01-02 起皆為生成輸出 |
| ticker 對齊 | ✓ 同一 `TICKERS` 順序 |

**但必須在論文中揭露的限制**：synthetic 序列**不是 free-running simulation**，
每 21 個交易日就被重新錨定到一段真實的 63 日 condition。
後果：**超過 21 個 lag 的時間相依性有一部分來自真實 condition，而非模型本身**。
這直接決定了 §PHASE 3 中 VC / LE 的 lag 上限 K = 20 < PRED_LEN = 21。

---

## PHASE 2 — Time Window Selection

**選擇準則（在看到任何模型分數之前固定）**：市場代表性、regime 多樣性、極端事件數量、
資料完整性、以及**每個 metric 的訊號是否高於其估計噪音**。

### 2.1 逐年市場環境（真實資料，10 檔平均）

| 年 | T | 年化波動 | \|r\|>3σ (%) | VC(ACF1 of r²) | LE(lag1) | Skew | ExKurt |
|---|---|---|---|---|---|---|---|
| 2015 | 252 | 0.246 | 1.15 | 0.078 | −0.100 | +0.116 | 5.32 |
| 2016 | 252 | 0.230 | 0.83 | 0.092 | −0.030 | −0.317 | 6.39 |
| 2017 | 251 | 0.172 | 0.72 | 0.012 | −0.004 | +0.651 | 12.54 |
| 2018 | 251 | 0.277 | 1.55 | 0.103 | −0.065 | −0.255 | 4.03 |
| 2019 | 252 | 0.220 | 0.79 | 0.129 | −0.103 | −0.348 | 4.81 |
| **2020** | 253 | **0.423** | **6.01** | **0.378** | −0.103 | +0.001 | 7.13 |
| 2021 | 252 | 0.233 | 0.67 | 0.051 | −0.013 | −0.474 | 7.87 |
| 2022 | 251 | 0.334 | 2.71 | 0.058 | −0.057 | −0.293 | 4.63 |
| 2023 | 250 | 0.245 | 0.88 | 0.030 | +0.006 | −0.386 | 7.21 |
| 2024 | 252 | 0.293 | 1.59 | **0.030** | **−0.018** | **−0.960** | **15.29** |

### 2.2 候選 windows

| | Window | 期間 | T | 市場背景 | 年化波動 | Hill α (k) | VC | LE | 3σ 事件% |
|---|---|---|---|---|---|---|---|---|---|
| **A** | 2024 only | 2024-01-02 ~ 2024-12-31 | 252 | AI 行情 + 8 月 VIX 尖峰 | 0.293 | 3.70 (k≈6) | 0.030 | −0.018 | 1.39 |
| **B** | COVID crash | 2020-01-02 ~ 2020-12-31 | 253 | 系統性危機 | 0.423 | 2.56 (k≈6) | 0.378 | −0.103 | 2.09 |
| **C** | 升息熊市 | 2022-01-03 ~ 2022-12-30 | 251 | 通膨 / 熊市 | 0.334 | 3.35 (k≈7) | 0.058 | −0.057 | 1.39 |
| **D** | Walk-forward 全段 | 2018-01-02 ~ 2024-12-31 | **1,761** | 平靜 → 危機 → 熊市 → 復甦，涵蓋 4 個 regime | 0.297 | 2.82 (k≈42) | 0.233 | −0.066 | 1.61 |
| **E** | 危機後 | 2021-01-04 ~ 2024-12-31 | 1,005 | 復甦 + 熊市 | 0.279 | 3.30 (k≈25) | 0.075 | −0.044 | 1.26 |

各 window 逐項評估：

- **A（2024）**：唯一涵蓋全部 32 個 Family A 模型的期間，**可比性最佳**。
  市場面也不算平淡（負偏度 −0.96、超額峰度 15.3 都是十年最極端）。
  但 T=252 造成三項致命問題（見 2.3）。
- **B（2020）**：extreme events 最豐富、VC 訊號最強（0.378），是評估 dynamics 的理想期間。
  但**只有 walk-forward 模型有輸出**，且單一 regime 不具代表性，作為主 window 會過度偏向危機情境。
- **C（2022）**：熊市代表性佳，但統計量與 A 類似地受 T=251 限制，且同樣只有 walk-forward 有資料。
- **D（2018–2024）**：regime 覆蓋最完整（含 2018Q4 修正、2020 COVID、2022 熊市、2023–24 復甦），
  T=1,761 使所有 metric 的估計誤差降到可接受範圍。**缺點：32 個 Family A 模型沒有這段輸出。**
- **E（2021–2024）**：折衷，但缺 COVID 這個唯一的系統性危機樣本，regime 代表性弱於 D。

### 2.3 決定性證據：2024 無法支撐五個 stylized facts 中的三個

以 stationary block bootstrap（Politis–Romano，block 長度 ≈ T^{1/3}）從**真實資料**重抽兩組
獨立樣本，計算兩組估計值之間的距離期望 τ —— 這是「即使模型完美，也必然存在的估計噪音尺度」。

| Metric | τ (2024, T=252) | τ (2018–24, T=1761) | 真實訊號大小 (2024) | 真實訊號大小 (2018–24) |
|---|---|---|---|---|
| Hill (log-ratio) | 0.228 | 0.110 | α = 3.70 | α = 2.82 |
| Skewness | **1.301** | 0.523 | **skew = −0.96** | skew = −0.45 |
| Jump frequency | 0.0159 | 0.0096 | f = 0.05（q95 定義） | f = 0.05 |
| VC curve | 0.047 | 0.050 | **curve mean = −0.008** | curve mean = **+0.092** |
| LE curve | 0.064 | 0.034 | **curve mean = −0.001** | curve mean = **−0.029** |

三個致命問題：

1. **Skewness**：2024 的噪音尺度 τ = 1.30 **大於**真實訊號本身 |−0.96|。
   任何模型只要 skew 落在 (−2.3, +0.3) 都會拿到接近滿分 → 無鑑別力。
2. **Volatility clustering**：2024 的真實 ACF(r²) 曲線平均值僅 **−0.008**，
   亦即該年**幾乎不存在波動聚集**。用一個本身就沒有該性質的期間去評估模型是否重現該性質，
   在方法論上不成立。2018–2024 為 +0.092，訊號約為噪音的 1.8 倍。
3. **Leverage effect**：2024 為 **−0.001**，同樣接近零；2018–2024 為 −0.029。
   這也正是 §23「近零分母」問題最嚴重的來源。

此外 T=252 下 Hill 的尾部樣本 k≈6（相對 SE 40%）、5% 尾部曝險僅 12 筆，
tail dependence 的估計標準誤約 0.14。

### 2.4 Primary Window 建議

```
PRIMARY_EVAL_START = "2018-01-02"
PRIMARY_EVAL_END   = "2024-12-31"        # T = 1,761
```

**理由（全部只依據真實資料，未涉及任何模型分數）**：

1. **唯一同時具備足夠訊號與足夠樣本的 window**：五個 stylized facts 的真實值全部顯著高於
   其自身的重抽噪音尺度（見 2.3 表），2024/2022/2020 皆做不到。
2. **regime 代表性最完整**：涵蓋平靜期（2019、2021）、系統性危機（2020）、
   政策性熊市（2022）、復甦期（2023–24），符合「synthetic market 是否在多種市場狀態下
   都保留統計性質」的研究目標。
3. **估計可靠度**：Hill 尾部樣本 k 從 6 提升至 42（相對 SE 40% → 15%）；
   5% 尾部曝險從 12 筆提升至 88 筆，使 tail dependence 可估。
4. **資料完整性**：10 檔在該期間零缺值。
5. **與 rolling generation 的設計一致**：walk-forward 的生成期間本身就是 2018-01-02 ~ 2024-12-31，
   evaluation window 等於生成期，不需要額外截斷，也不引入任何未來資料。

**Anti-cherry-picking 聲明**：本 window 的選擇僅使用真實市場資料的統計性質
（波動率、極端事件數、ACF 訊號 vs 噪音比、尾部樣本數），
**在計算任何模型的 fidelity 分數之前即已固定**。
選擇準則與任何模型的表現無關，且刻意排除了「模型表現最好的期間」這一類理由。

### 2.5 由此產生的重大取捨（需要使用者裁決）

Primary window = 2018–2024 意味著 **32 個 Family A 模型（只有 2024 輸出）無法進入主表**。
三個選項：

- **選項 1（推薦）**：以 walk-forward 協定重新生成 Family A 中要進入論文的那幾個配置
  （例如 `dow10_sty1_c`、`dow10_eps`、`dow10_studt_df3`、`dow10_gpd_l20`、`dow10_abl_base`），
  取得 2018–2024 輸出後，全部模型在同一 window 下比較。
  這是唯一能讓主表既統計可靠又完全可比的作法。**需要 GPU 重跑生成（不在本 notebook 範圍）。**
- **選項 2**：主表用 walk-forward 模型（2018–2024），Family A 另立一張
  **明確標註 window = 2024、T=252、且標註三個 metric 統計功效不足**的次表，兩表不合併排名。
- **選項 3**：接受 2024 作為 primary window，換取全模型可比，但必須在論文中聲明
  Skew / VC / LE 三項在該 window 下不具鑑別力。**不建議**。

Notebook 會把 window 寫成 configuration，三個選項都只需改參數。

---

## PHASE 3–6 — Metric / Distance / Score / Weighting 設計

### 3.0 統一的 normalization：以「真實資料自身的重抽噪音」為尺度

所有五個 stylized facts 與三個 cross-asset metric 共用同一個轉換：

```
Score_m = 2^( − D_m / τ_m )                     ∈ (0, 1]
```

其中 `τ_m` = 由**真實資料**的 stationary block bootstrap 得到的
「兩組獨立真實重抽之間的期望距離」：

```
τ_m = E_b [ D_m( θ̂(X_real^{(b1)}), θ̂(X_real^{(b2)}) ) ] ,  b = 1…B  (B = 1000)
```

**為什麼這個 normalization 不是任意的**：

| 要求 | 滿足方式 |
|---|---|
| D=0 → Score=1 | `2^0 = 1` ✓ |
| 距離越大 → 越接近 0 | 單調遞減、以 0 為下確界 ✓ |
| 不受 metric 本身 scale 影響 | D 以 τ 為單位，τ 與 D 同單位，比值無量綱 ✓ |
| 不使用 synthetic 決定門檻 | τ 只用真實資料 ✓（避免 circularity） |
| 近零分母問題（§23） | **不除以 metric 值，而是除以其標準誤**。LE 真實值 −0.029、τ=0.034，比值良好；若改用相對誤差則會爆炸 ✓ |
| 可解釋性 | **D = τ ⟺ Score = 0.5**：模型誤差恰等於真實估計量自身的重抽誤差時得 0.5 分 ✓ |

替代方案比較：

- `1 − D`：需要 D ∈ [0,1]，只有 TV distance 滿足，其餘 metric 不成立 → 否決。
- `exp(−D)`：τ 隱含被設為 1/ln2，是任意的 → 否決。
- 相對誤差 `D/|θ_real|`：在 LE、Skew、Jump frequency 上分母近零 → 否決（正是 §23 的問題）。
- Gaussian kernel `exp(−½(D/τ)²)`：可接受的替代；比 `2^(−D/τ)` 對小誤差更寬容、對大誤差更嚴厲。
  Notebook 會把 kernel 形式做成參數，主表用 `2^(−D/τ)`，附錄報 Gaussian 版的排名穩定性。

### 3.1 Heavy Tail

| 步驟 | 定義 |
|---|---|
| Raw metric | 左尾 Hill 指數。令 X = 由大到小排序的 {−r : r < 0}，k = ⌈n_neg^{2/3}⌉，u = X_(k)：<br>`α̂ = [ (1/k) Σ_{j=1..k} log(X_(j)/u) ]^{-1}` |
| Real / Syn | `α_real,i`、`α_syn,i`，i = 1…10 逐檔 |
| Raw distance | `D_tail,i = |log(α_syn,i / α_real,i)|` |
| Score | `S_tail,i = 2^(−D_tail,i / τ_tail)`，τ_tail = 0.110（2018–24） |

**為什麼用 log-ratio 而非 `|Δα|/α_real`**：α 是正的尺度參數，其估計量近似對數常態。
log-ratio 對「高估 2 倍」與「低估 2 倍」給出相同距離（`|log 2| = |log ½|`），
而相對誤差會給 1.0 vs 0.5 —— 不對稱地偏袒低估。此外 α>0 有下界，
不存在近零分母問題，故 log 變換安全。

理論方向確認：`P(X > x) ~ x^(−α)`，α 小 → 尾巴厚。✓

**穩健性**：同時報 k ∈ {⌈n^{2/3}⌉/2, ⌈n^{2/3}⌉, 2⌈n^{2/3}⌉} 三組結果（Hill plot 敏感度）。

### 3.2 Negative Skewness

| 步驟 | 定義 |
|---|---|
| Raw metric | `g₁ = m₃/m₂^{3/2}`，用 `scipy.stats.skew(..., bias=False)` |
| Raw distance | `D_skew,i = |skew_syn,i − skew_real,i|`（**絕對距離**，因 skew 無量綱且可跨零，相對距離無定義） |
| Score | `S_skew,i = 2^(−D_skew,i / τ_skew)`，τ_skew = 0.523（2018–24） |
| **Sign agreement（獨立呈現）** | `A_skew = (1/10) Σ_i 1{ sign(skew_syn,i) = sign(skew_real,i) }` |

**為什麼把 magnitude 與 sign 分開而不合成一個懲罰項**：合成需要一個懲罰係數 γ，
而 γ 沒有理論依據 → 違反「禁止任意 normalization」。
實際上絕對距離已經隱含懲罰了符號錯誤（real = −0.95、syn = +0.10 → D = 1.05 ≈ 2τ → Score ≈ 0.25），
所以不需要額外機制。Sign agreement 以**獨立欄位**呈現，回答的是不同的問題：
「模型是否至少抓到下跌尾部較厚這個方向性」。

### 3.3 Jump Behaviour

| 步驟 | 定義 |
|---|---|
| **Threshold（由真實資料定義）** | `u_i = q_{0.95}( |r_real,i| )`，逐檔，於 primary window 上計算。**synthetic 一律套用此 u_i** |
| Frequency | `f_i = #{ t : |r_t,i| > u_i } / T`（real 依定義 = 0.05） |
| Magnitude | `m_i = E[ |r_t,i| given |r_t,i| > u_i ]`；若無超越事件 → 記為 `no exceedance`，magnitude score = 0（不可 NaN-drop） |
| `D_jump_freq,i` | `|f_syn,i − f_real,i|`；τ 用**二項標準誤的閉式解** `τ_f = √(2·f_real(1−f_real)/T)`（兩組獨立估計之差的 SD） |
| `D_jump_mag,i` | `|log(m_syn,i / m_real,i)|`（正的尺度參數 → log） |
| 合成 | `D_jump,i` 不直接合成；先各自轉分數再平均：<br>`S_jump,i = ½·2^(−D_freq,i/τ_f) + ½·2^(−D_mag,i/τ_m)` |

**為什麼 q95 而非 q99**：門檻選擇只依據**曝險數**，不依據結果。
q99 在 T=1761 下每檔約 18 筆、在 T=252 下僅 2.5 筆；q95 分別為 88 筆與 12.6 筆。
主表用 q95（88 筆，二項 SE ≈ 0.005）；附錄報 q99 的敏感度。

**為什麼 frequency 與 magnitude 等權**：複合 Poisson 跳躍模型
`r_t = σ dW_t + J_t dN_t` 的跳躍成分由兩個參數完全刻畫 —— 強度 λ（→ frequency）
與跳躍幅度分布（→ magnitude）。沒有理論說其中一個更重要，故等權，
與 §16「無充分理由則等權」一致。

**先在 raw distance 層合成、還是先轉分數再合成？**
選後者。frequency 是比例（單位 = 機率）、magnitude 是對數比（無量綱），
直接加權平均會把不同單位的量混在一起 —— 正是 §33.12 禁止的事。

**必須揭露的結果**：由於多數模型有嚴重的變異數收縮
（VolRatio 0.22–0.57），在真實 q95 門檻下**多數模型的 jump frequency 會趨近 0**。
這是**真實的評估結果，不是 bug** —— 它正確地指出「這些模型無法產生真實幅度的極端事件」。
為了保留鑑別力，另外報一個**診斷欄**：對標準化報酬 `z = r/σ_r`（各自的 σ）
重做同一計算，回答「在尺度校正之後，跳躍的**形狀**是否正確」。
主表用絕對門檻版（符合 §7 反 circular 要求），標準化版僅作為診斷。

### 3.4 Volatility Clustering

| 步驟 | 定義 |
|---|---|
| Proxy | `r²`（而非 \|r\|；r² 對應 realized variance，與 BNS/GARCH 文獻一致） |
| Raw metric | `ρ_i(k) = Corr(r²_t, r²_{t+k})`，k = 1…K，**K = 20** |
| Raw distance | `D_VC,i = (1/K) Σ_{k=1..K} |ρ_syn,i(k) − ρ_real,i(k)|`（L1 曲線距離） |
| Score | `S_VC,i = 2^(−D_VC,i / τ_VC)`，τ_VC = 0.050（2018–24） |

**為什麼 K = 20**（三個獨立理由）：

1. **經濟意義**：約一個交易月，是波動聚集在實務上（風險模型、選擇權定價）最相關的水平。
2. **統計意義**：Bartlett 標準誤 ≈ 1/√T；T=1761 → 0.024。真實 ACF(r²) 在 k>20 後
   已逼近該噪音帶，再往後只是在比較噪音。
3. **生成協定的硬限制（最關鍵）**：walk-forward 的 `PRED_LEN = 21`，
   每 21 日重新錨定到真實 condition。**lag ≥ 21 的相依性有一部分來自真實 condition
   而非模型**，納入評估會系統性地高估模型。故必須 K < 21。

**為什麼 L1 曲線距離**：與逐點比較相比，L1 平均對單一 lag 的離群值穩健；
與 L2 相比，L1 不會讓某一個 lag 的大偏差主導整條曲線。
替代方案（附錄報告）：以 `1/k` 加權（前期 lag 權重較高），
但缺乏理論依據指定衰減率，故主表不採用。

**為什麼這能代表 volatility clustering**：clustering 的定義就是
「波動的自相關為正且緩慢衰減」；比較整條 ACF 曲線同時檢驗了
**水平**（clustering 有多強）與**衰減形狀**（記憶有多長），
而單一 lag-1 只檢驗前者。

### 3.5 Leverage Effect

| 步驟 | 定義 |
|---|---|
| Raw metric | `L_i(k) = Corr(r_t, r²_{t+k})`，k = 1…K，K = 20（理由同 3.4） |
| Raw distance | `D_LE,i = (1/K) Σ_k |L_syn,i(k) − L_real,i(k)|` |
| Score | `S_LE,i = 2^(−D_LE,i / τ_LE)`，τ_LE = 0.034（2018–24） |
| **方向診斷（獨立呈現）** | (a) `sign( mean_k L_syn,i(k) )` 是否為負；(b) 逐 lag 符號一致率 `(1/K)Σ_k 1{sign match}` |

真實資料 2018–2024 的 `mean_k L_real(k) = −0.029`，符合金融文獻中的負向關係
（下跌 → 未來波動上升）。方向與數值距離分開報告，理由同 3.2：
避免引入沒有依據的符號懲罰係數。

**近零問題處理**：`|L_real| = 0.029` 與 `τ_LE = 0.034` 同一量級，
這正是為何**絕不可**使用 `D/|L_real|` 作為 normalization。
以 τ 為尺度時，「真實值本身就在噪音邊緣」這個事實被誠實地反映為
「此 metric 的鑑別力有限」，而不是被放大成假的巨大相對誤差。
Notebook 會為每個 metric 報 **signal-to-noise ratio `|θ_real| / τ`**，
讓 reviewer 直接看到哪些 metric 具有統計功效。

### 3.6 逐檔聚合（§10）

```
for i in TICKERS:            # 嚴禁先 concatenate 10 檔再算
    θ_real,i , θ_syn,i  →  D_m,i  →  S_m,i
S_m = (1/10) Σ_i S_m,i
```

per-ticker 的 `θ_real,i`、`θ_syn,i`、`D_m,i`、`S_m,i` 全部保存為
`results/tables/asset_level_<model>.csv`，供 appendix 與 Table SF-3 使用。

**為什麼不 concatenate**：把 10 檔串接會製造 9 個假的「跨股票斷點」，
污染所有時序 metric（VC、LE）；同時把不同股票的尺度混在一起，
使 Hill 與 skewness 變成混合分布的統計量而非單一資產的性質。

### 3.7 Stylized Facts 權重（§11）

兩個方案：

```
(a) Equal:         SF = (1/5)(S_HT + S_Skew + S_Jump + S_VC + S_LE)

(b) Hierarchical:  Distribution = ½ S_HT + ½ S_Skew
                   Dynamics     = ⅓ S_Jump + ⅓ S_VC + ⅓ S_LE
                   SF = 0.5 · Distribution + 0.5 · Dynamics
                   （等效權重 0.25, 0.25, 0.167, 0.167, 0.167）
```

**建議採用 (b) hierarchical**，理由：

在等權方案下，Dynamics 類別得到 60% 的總權重、Distribution 只有 40% ——
但這個 60/40 純粹來自「我們碰巧定義了 3 個 dynamics metric 和 2 個 distribution metric」，
是**指標數量的產物，不是金融意義的判斷**。
Hierarchical 讓權重反映「分布性質」與「時序/事件動態」這兩個概念軸的
對等地位，而不是各軸下我們選了幾個代理指標。
這也讓框架在未來新增指標時保持穩定（多加一個 dynamics metric 不會偷偷改變兩軸的相對重要性）。

主表用 (b)，附錄同時報 (a) 的排名，證明結論不依賴權重選擇。
**權重在看到任何模型分數之前固定。**

### 3.8 Cross-Asset Metric 1：Correlation Matrix

| 步驟 | 定義 |
|---|---|
| Raw metric | Pearson 相關矩陣 `C ∈ R^{10×10}`（real 與 syn 各一） |
| Raw distance | `D_corr = (2 / (N(N−1))) Σ_{i<j} |C_real(i,j) − C_syn(i,j)|`（45 個上三角元素，排除對角線） |
| Score | `S_corr = 2^(−D_corr / τ_corr)`，τ_corr 由真實資料 block bootstrap 得出 |

**normalization 是否需要**：D_corr 理論上界為 2（相關係數皆在 [−1,1]），
但實務上兩個真實重抽之間的距離約 0.02–0.05，所以「除以 2」會讓所有分數擠在 0.99 附近，
毫無鑑別力。以 τ 為尺度才是有意義的 normalization。

**穩健性**：同時報 Spearman 版本（報酬非常態，Pearson 受極端值影響）。

### 3.9 Cross-Asset Metric 2：PCA / Eigenvalue Spectrum

| 步驟 | 定義 |
|---|---|
| 矩陣選擇 | **相關矩陣**（非共變異矩陣） |
| Raw metric | 特徵值 λ₁ ≥ … ≥ λ₁₀，`EVR_k = λ_k / Σλ`（相關矩陣下 Σλ = 10） |
| Raw distance | `D_PCA = ½ Σ_{k=1..10} |EVR_real,k − EVR_syn,k|`（**total variation distance**，∈ [0,1]） |
| Score | `S_PCA = 2^(−D_PCA / τ_PCA)` |

**為什麼用相關矩陣而非共變異矩陣**（兩個理由）：

1. 共變異矩陣的特徵值會被**變異數尺度差異**主導。已知多數模型 VolRatio 只有 0.22–0.57，
   用共變異矩陣等於把「尺度錯誤」重複計入 cross-asset 分數 —— 而尺度錯誤已經在
   univariate 層（Hill、Jump magnitude）被評估過。**重複計分會扭曲 Overall Score。**
2. 金融 RMT / 因子結構文獻（Laloux et al. 1999；Plerou et al. 2002）標準做法即為相關矩陣，
   使結果可與既有文獻對照。

**為什麼 TV distance 而非單純 mean absolute difference**：
EVR 是機率向量（兩者皆和為 1），兩個機率向量的 `½Σ|Δ|` 正是 total variation distance，
**天然有界於 [0,1]** 且有明確機率意義（「兩個 spectrum 在最壞情況下的可區分程度」）。
單純 mean 只是差了常數 1/20，但失去這個解釋。

**PCA 在此的角色**（依 §14 要求明寫）：**不是用來預測**，
而是檢驗 synthetic market 是否保留真實市場的**共同潛在因子結構** ——
λ₁ 對應「市場因子」的強度，後續特徵值的衰減形狀對應產業/風格因子的分散程度。
一個只學到邊際分布、卻讓 10 檔股票近似獨立的模型，會呈現接近均勻的 spectrum（EVR_k ≈ 0.1），
與真實市場的 λ₁ 主導形態明顯不同。

**額外診斷欄**：`EVR₁_real` vs `EVR₁_syn`（市場因子強度），單獨報告。

**冗餘性（§12 要求檢查）**：`S_corr` 與 `S_PCA` **並非獨立** ——
EVR spectrum 是相關矩陣的函數，且 λ₁/N ≈ 平均相關係數。
兩者的差別在於：Corr-MAE 是**逐元素**距離（哪一對股票錯了），
EVR 是**譜**距離（整體因子集中度）。
兩個相關矩陣可以有相同的平均相關但完全不同的特徵值分布（例如 block 結構 vs 均勻結構）。
**處理方式**：保留兩者，但在 Robustness section 明確報告
兩個 score 在所有模型上的實證相關係數；若 |corr| > 0.9，
則在論文中聲明它們提供的是同一項資訊，並將 Cross-Asset 權重改為
`S_corr` 與 `S_PCA` 合計 1/2、`S_tail` 1/2（避免相關結構被雙重計分）。
此規則**事先設定**，不依模型結果調整。

### 3.10 Cross-Asset Metric 3：Lower Tail Dependence

| 步驟 | 定義 |
|---|---|
| Raw metric | `λ_L(i,j) = P( U_j < u | U_i < u )`，u = 0.05，U = 各序列的經驗 CDF（rank transform） |
| 矩陣 | 10×10，對角線為 1，取 90 個非對角元素（或對稱化後 45 對） |
| Raw distance | `D_tail = (1/|P|) Σ_{(i,j) ∈ P} |λ_L,real(i,j) − λ_L,syn(i,j)|` |
| Score | `S_tailDep = 2^(−D_tail / τ_tailDep)` |

**為什麼用下尾相依而非單純相關**：
Pearson 相關是**平均**的線性共動，對常態分布是完整刻畫，但真實報酬非常態。
金融危機的本質是**尾部共動**：平時相關 0.4 的兩檔股票，在崩盤時可能同時跌 —— 
這是投資組合分散化失效的機制，也是 VaR / CVaR 模型最關鍵的輸入。
相關矩陣完全看不到這件事（兩個相關矩陣相同的市場，可以有完全不同的尾部相依）。
對 portfolio 風險而言，`λ_L` 直接決定「同時發生大額虧損」的機率。

**與 §15 的衝突與建議（需要使用者裁決）**：

§15 要求「用 real data 定義的 threshold，不要 real 和 synthetic 各自取 q05」。
量化檢查的結果：
以真實 2024 的 q05 為門檻，**32 個模型中有 24 個的平均曝險數 < 5 筆**
（真實為 12.6 筆），其中 20 個為 **0 筆** → 條件機率 `0/0` **無定義**，
指標對多數模型完全崩潰。根本原因是嚴重的變異數收縮（VolRatio 0.22–0.57）。

方法論上，尾部相依係數 `λ_L` 依定義是一個 **copula 泛函**，
定義在 `U = F(X)` 上，對邊際分布的單調變換不變。
使用各自的經驗分位數**不是**「各自挑對自己有利的門檻」，
而是該量的**標準定義**（Joe 1997；McNeil–Frey–Embrechts 2015）。
邊際尺度的錯誤已經在 univariate 層（Hill、Jump magnitude、以及獨立報告的 VolRatio）被懲罰過，
在此重複計入會**雙重計分**。

**建議**：
- **主指標**：copula 版（各自 rank transform，同一個機率水平 u = 0.05）→ 度量**相依結構**。
- **輔助指標**：absolute 版（真實 q05 門檻）→ 度量**絕對尾部共同曝險**，
  對曝險為 0 的模型記為 `undefined (0 exceedances)` 並明確標示，**不以 0 分混入平均**。
- 兩者都報，並在 methodology report 中說明兩者回答不同問題。

若使用者堅持 §15 原文，則主表改用 absolute 版，但必須接受 24/32 模型該欄為 undefined。

### 3.11 Cross-Asset 合成權重（§16）

```
CrossAsset = (1/3)(S_corr + S_PCA + S_tailDep)
```

等權，理由：三者分別對應**線性共動水平**、**共同因子結構**、**極端共動**，
沒有理論依據認定何者更重要。唯一的例外是 3.9 事先設定的冗餘規則。

### 3.12 Overall Score（§17）

```
Overall = 0.5 · SF_Score + 0.5 · CrossAsset_Score
```

採 50/50，理由：univariate fidelity 與 multivariate fidelity 是兩個
**互補且不可互相取代**的品質軸 —— 一個邊際分布完美但股票間互相獨立的生成器，
與一個相關結構完美但邊際分布是高斯的生成器，作為 synthetic market 都同樣不可用。
沒有充分理由偏離對等，故不偏離。

### 3.13 統計可靠度（§24）

| Metric | 方法 | 說明 |
|---|---|---|
| Hill α | **subsampling / m-out-of-n bootstrap** | 一般 bootstrap 對極值順序統計量**不一致**（Athreya 1987），不可直接用 |
| Skewness | stationary block bootstrap | 高階動差，收斂慢，CI 會很寬（誠實反映） |
| Jump frequency | **二項閉式 CI**（Wilson score interval） | 比例的精確方法，優於 bootstrap |
| Jump magnitude | block bootstrap，條件於超越事件 | 超越事件 < 10 筆時標記為不可靠 |
| VC / LE curve | stationary block bootstrap，block L ≈ T^{1/3} | 保留時序相依 |
| Corr / PCA | block bootstrap（整個 10 維向量一起重抽，保留橫斷面相依） | 不可逐檔獨立重抽 |
| Tail dependence | block bootstrap + 曝險數檢查 | 曝險 < 20 時標記不可靠 |
| **Walk-forward 模型額外** | **跨 50 條路徑的離散度** | 這是**生成不確定性**，與估計不確定性不同，須分開報告 |

每張 score 表都附 bootstrap 95% CI；signal-to-noise ratio `|θ_real|/τ` 一併報告。

---

## 需要新增 / 修改的函式清單

**新增 module `evaluation/` （不動任何既有生成程式碼）**

| 檔案 | 函式 | 說明 |
|---|---|---|
| `loaders.py` | `discover_models()` | 掃描 `outputs/dow10/walk_forward/{timediff,statistical,external}/` 與 `outputs/dow10/single_shot/`（相容舊平面目錄），依檔案樣式分類 Family A/B/C，回傳 `MODELS` |
| | `load_real(start, end)` | 讀 `dow10_wf_returns.csv`，回傳 (T,10) rclose |
| | `load_synthetic(model, start, end)` | 統一 preprocessing layer：Family A 補 date、Family B 篩日期、Family C 排除 |
| | `assert_alignment(real, syn)` | 日期、ticker、長度三重檢查 |
| `metrics_uni.py` | `hill_alpha(r, k=None)` | 左尾，k-based |
| | `skewness(r)` | `bias=False` |
| | `jump_freq_mag(r, u)` | 外部給定 threshold |
| | `acf_r2_curve(r, K)` | VC 曲線 |
| | `leverage_curve(r, K)` | LE 曲線 |
| | `bns_relative_jump(r)` | **未裁切**，僅診斷 |
| `metrics_cross.py` | `corr_matrix(X, method)` | Pearson / Spearman |
| | `evr_spectrum(X)` | 相關矩陣特徵值 → EVR |
| | `lower_tail_dep_copula(X, u)` | rank-based |
| | `lower_tail_dep_absolute(X, thr_real)` | 真實門檻版，回傳曝險數 |
| `calibration.py` | `stationary_block_bootstrap(x, L, rng)` | Politis–Romano |
| | `tau_noise_floor(real, metric_fn, dist_fn, B)` | 計算 τ（只用真實資料） |
| | `score(D, tau, kernel="halving")` | `2^(−D/τ)` / Gaussian |
| `aggregate.py` | `per_ticker_scores(...)` | 逐檔 → 保存 |
| | `sf_score(...)`, `ca_score(...)`, `overall(...)` | 階層加權 |
| `report.py` | `save_figure_with_data(fig, name, data)` | 同時輸出 png + txt 數值 |

**沿用（不改）**：`eval_p0p1.py:37 corr_mae` 的上三角邏輯（重寫進 `metrics_cross.py`）。

**明確不重寫**：`generate_dow10.py` 的 `compute_stylized_facts` 保留原樣供既有腳本使用，
evaluation notebook 不呼叫它（避免動到生成流程的相依）。

---

## 已定案的決定（使用者裁決）

1. **Primary window：採 Option 1。** `PRIMARY_EVAL_START = 2018-01-02`、
   `PRIMARY_EVAL_END = 2024-12-31`。所有要進入論文主表的代表性模型，都必須使用**相同的
   rolling generation protocol** 重新取得完整輸出；只有 2024 單年輸出的配置不得進入主表。
2. **主表形式：Method × Time。** 2018–2024 是唯一決定排名的 Primary Evaluation；
   2020 Crisis 與 2022 Bear 僅作 regime robustness analysis。某個 regime 若無法公平比較，
   標 N/A 並在報告中說明，不硬填分數。
3. **Tail dependence：copula / rank-based 為主指標**，absolute-threshold 版保留為
   supplementary analysis（不刪除）。Main metrics = Correlation Matrix Distance、
   PCA / Eigenvalue Structure Distance、Copula-based Tail Dependence Distance。
4. **模型範圍：** 論文最終只會有兩個版本 —— baseline 與 fine-tune 完成的模型。
   既有 32 個單次 2024 生成目錄暫不納入。目前 baseline = `outputs/dow10/walk_forward/timediff/dow10_wf_v2`
   （28 cycles、50 paths、vol_scale 1.0、gen_seed 1）。

---

## 實作狀態（PHASE 7–13 已完成）

| 產出 | 路徑 |
|---|---|
| Notebook（20 sections + audit，已執行） | `evaluation_framework.ipynb` |
| Evaluation package | `evaluation/{config,loaders,metrics_uni,metrics_cross,calibration,pipeline,report,run_all}.py` |
| 表格 | `results/tables/*.csv` |
| 圖表（png + 數值 txt） | `results/figures/`、`results/figure_data/` |
| Methodology report | `results/evaluation_methodology_report.txt` |
| Paper results summary | `results/paper_results_summary.txt` |

重跑方式：`python -m evaluation.run_all`（約 130 秒，B=500）。
加入 fine-tuned 模型只需把它的 tag 加進 `evaluation/config.py` 的 `MODELS_WHITELIST`。
