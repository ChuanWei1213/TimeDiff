# Problem Setting / 問題設定

**Scenario-Conditioned Diffusion Generation of Multivariate Financial Time Series
with Application to ETF Constituent-Driven Portfolio Weight Transition**

**以情境為條件的多變量金融時間序列擴散生成 —— 應用於 ETF 成份股變動之投資組合權重轉換**

---

# 中文版

## 1. 研究背景

### 1.1 動機：單一歷史軌跡的限制

金融應用（風險管理、投資組合最佳化、強化學習交易代理人）長期受限於**單一歷史軌跡**：真實市場只發生過一次，可觀測的極端事件樣本極少。以本研究採用的 Dow 10 為例，2018–2024 共 1,761 個交易日，每檔股票落在 5% 下尾門檻以下的事件僅約 **88 筆**。任何需要估計尾部風險、或需要大量情境樣本的方法，都會受制於此。

以生成模型合成金融時間序列，可提供任意數量的市場路徑，是突破此限制的自然途徑。而在各類生成模型中，**conditional diffusion model** 因其訓練穩定、且能以外生資訊 $c$ 導引生成，成為此任務的主流選擇。

### 1.2 基礎模型：TimeDiff

本研究以 **TimeDiff**（Shen & Kwok, ICML 2023, *Non-autoregressive Conditional Diffusion Models for Time Series Prediction*）為基礎架構。TimeDiff 是一個**非自迴歸的條件擴散模型**，其設計特點如下：

**(a) 非自迴歸的區塊生成。** 不同於逐點遞迴生成，TimeDiff 一次生成長度為 $H$ 的整個區塊 $X = (r_{t+1},\dots,r_{t+H})$，避免了自迴歸模型的誤差累積。本研究設定 $H = 21$（約一個交易月）。

**(b) 雙分支條件編碼。** 條件資訊（過去 $L=63$ 日的多變量序列）經兩條路徑進入模型：

- **Per-variable 線性投影**：對每個變數獨立地由歷史視窗線性映射到未來 $H$ 日
- **CNN 條件網路**：跨變數的卷積編碼，捕捉橫斷面資訊

兩者相加得到一個「條件先驗預測」$\hat{X}_{\text{cond}}$。

**(c) Future Mixup。** 訓練時，上述條件先驗預測會與**真實未來** $X$ 以逐元素隨機比例混合：

$$\tilde{X}_{\text{cond}} = m \odot \hat{X}_{\text{cond}} + (1-m) \odot X, \qquad m \sim \mathcal{U}(0,1)^{H\times N}$$

此設計類似 scheduled sampling，目的是穩定條件對齊。推論時 $\tilde{X}_{\text{cond}} = \hat{X}_{\text{cond}}$（無真實未來可用）。

**(d) 去噪網路輸入。** CNN denoiser 接收三者的串接：

$$\text{input} = \left[\,\underbrace{x_s}_{\text{加噪後的目標}},\;\; \underbrace{\tilde{X}_{\text{cond}}}_{\text{條件先驗預測}},\;\; \underbrace{\text{DLinear}(c^{\text{hist}})}_{\text{短期線性外推}}\,\right]$$

並額外接收雜訊層級 $s$ 的 embedding。

**(e) 取樣。** 以 DPM-Solver（機率流 ODE，二階多步）在 $S=100$ 的離散化上取 20 步完成反向生成，取樣器對雜訊層級的排程為 cosine schedule。

TimeDiff 之所以適合本研究，關鍵在於**它原生支援條件輸入** —— 這正是後續要編入「金融時序知識」與「目標情境」的介面。

### 1.3 資料與標的

- **標的**：Dow 10 成份股（AAPL, AMGN, CRM, CSCO, IBM, INTC, MSFT, NKE, VZ, WMT）
- **特徵**：每檔 3 個通道 —— $r^{\text{close}}_t = \log(C_t/C_{t-1})$（跨日報酬）、$r^{\text{high}}_t = \log(H_t/C_t)$、$r^{\text{low}}_t = \log(L_t/C_t)$（日內區間），共 $N \times 3 = 30$ 個通道
- **stylized facts 僅定義於 $r^{\text{close}}$ 通道**（跨日報酬），日內區間通道不參與統計性質的評估

## 2. Walk-Forward 評估協定

金融時間序列生成的評估極易受到資料洩漏污染。本研究建立一套嚴格的滾動重訓協定，其設計目的是**確保任一時點的生成，只使用該時點之前可取得的資訊**。

### 2.1 時間切分

| 參數 | 值 | 說明 |
|---|---|---|
| 生成期間 | 2018-01-02 ~ 2024-12-31 | 1,761 個共同交易日 |
| 重訓間隔（stride） | 63 個交易日 | 約一季 |
| 期數 | 28 期 | 最後一期為 60 日 |
| 每期訓練窗長度 | 1,260 個交易日 | 約 5 年 |
| Condition 長度 $L$ | 63 日 | 約一季 |
| 生成區塊長度 $H$ | 21 日 | 約一個月 |

第 $k$ 期的 origin 定義為生成期的第 $63k$ 個交易日。

### 2.2 每期的兩階段訓練

**Stage A — 選模。** 訓練窗前 1,008 日作訓練、後 252 日作驗證；scaler 僅以前 1,008 日 fit。每隔固定步數以驗證 MSE 評估，選出最佳 step $k^\star$。

**Stage B — 正式訓練。** **完全重新初始化模型與 scaler**（不 warm-start），以完整 1,260 日訓練至 step $k^\star$。

兩階段皆以 optimizer step（而非 epoch）為單位推進，因為兩階段的每 epoch step 數不同（925 vs 1,177 個訓練窗口）；學習率排程與各項 loss 的 annealing 亦以 step 為基準，確保「同一個 step」在兩階段具有相同的超參數狀態，選出的 $k^\star$ 才有意義。

### 2.3 無前視的保證

協定中有四道機制確保無資料洩漏：

1. **訓練窗嚴格早於 origin**：訓練資料取 $[\text{origin}-1260,\ \text{origin})$，程式中以 `assert` 強制檢查訓練窗最後一日 $<$ origin
2. **Scaler 只以訓練段 fit**，驗證段與生成期均以該 scaler 轉換
3. **Condition 只取當時可得的真實歷史**：第 $k$ 期第 $j$ 個區塊的條件為 $\text{real}[\,\text{origin}+21j-63 : \text{origin}+21j\,]$
4. **不把生成資料放回訓練窗**：每期訓練資料一律為真實資料

### 2.4 生成與價格重建

每期生成 $63 = 21 \times 3$ 日，分三個區塊，每區塊的 condition 皆為當時可取得的真實 63 日。

全期 1,761 日的報酬串接後，價格由**單一種子**遞迴重建：

$$C_t = C_{t-1}\exp\!\left(r^{\text{close}}_t\right), \qquad C_0 = \text{2017 年最後一個共同交易日的真實收盤價}$$

**區塊邊界與重訓邊界均不重置** —— 亦即第 $t$ 日的價格接的是第 $t-1$ 日的**生成**價格，而非真實價格。此設計確保價格軌跡是模型的完整產物。

### 2.5 評估協定的限制（須明確揭露）

生成序列**並非 free-running simulation**：每 21 個交易日就會被重新錨定到一段真實的 63 日 condition。因此：

- 超過 21 個 lag 的時間相依性，有一部分來自真實 condition 而非模型本身
- 所有時序類指標（波動聚集、leverage effect）的 lag 上限必須設為 $K = 20 < H = 21$，否則量到的是協定假象
- 本評估**不能**推論模型的長期自主模擬能力

## 3. 前期實證診斷

我們以上述協定對 TimeDiff 進行系統性診斷，得到兩項發現，構成本研究的問題起點。

### 3.1 發現一：生成過程退化為確定性映射

**症狀。** 生成報酬的標準差僅為真實的 **22.9%**；真實 5% 下尾門檻之下的曝險由 88 筆降至 **0.32 筆**。模型幾乎不產生真實幅度的極端事件。

**確認。** 固定同一 condition、以 32 組不同的擴散噪聲各生成一次，將輸出變異拆解為

$$\text{noise\_frac} = \frac{\mathrm{Var}_k(s)}{\mathrm{Var}_k(s) + \mathrm{Var}_{t,i}(\mu)}, \qquad \mu = \tfrac1K\textstyle\sum_k s_k$$

實測 **noise_frac $= 0.045$** —— 僅 4.5% 的輸出變異來自擴散噪聲，其餘 95.5% 由 condition 決定。模型實質上是一個確定性回歸器。

**根因定位。** 在低雜訊步 $s$ 時，$x_s = \sqrt{\bar\alpha_s}X + \sqrt{1-\bar\alpha_s}\varepsilon \approx X$，MSE 的最佳預測為 $E[X\mid x_s] = \sqrt{\bar\alpha_s}\,x_s$，對 $X$ 迴歸的係數即為 $\bar\alpha_s$。實測 denoiser 的響應斜率：

| $s$ | $\bar\alpha_s$（理論最佳斜率） | 實測斜率 | 實測/最佳 |
|---|---|---|---|
| 1 | 0.9983 | 0.0090 | **0.90%** |
| 10 | 0.9667 | 0.0092 | 0.95% |
| 40 | 0.6325 | 0.0074 | 1.17% |
| 95 | 0.0039 | 0.0004 | 10.3% |

在 $s=1$（輸入 99.8% 為原始資料、答案幾乎寫在輸入上）時，網路的響應僅為應有的 **0.9%**。以完美預測器 $\sqrt{\bar\alpha}x_s$ 代入同一量測程序可還原理論值（0.9979 vs 0.9983），確認量測本身無偏誤。

**因果鏈。** 擴散噪聲進入模型的唯一入口即為 $x_s$。網路不使用 $x_s$ $\Rightarrow$ 噪聲資訊無法傳遞 $\Rightarrow$ 輸出退化為 condition 的函數 $\Rightarrow$ noise_frac 崩潰。**單步的斜率失效是因，全鏈的 collapse 是果。**

### 3.2 第一層修正：v-parameterization

**問題本質。** $X$-prediction（原始設定）下，網路必須**自行從卷積層學出恆等映射**才能把 $x_s$ 的資訊送到輸出。而在近乎不可預測的報酬序列上，「輸出一個由 condition 決定的平滑均值」是 MSE 意義下的次佳捷徑，網路遂選擇了它。

**解法。** 改採 v-parameterization（Salimans & Ho, 2022）：

$$v = \sqrt{\bar\alpha_s}\,\varepsilon - \sqrt{1-\bar\alpha_s}\,X, \qquad \hat{X} = \underbrace{\sqrt{\bar\alpha_s}\,x_s}_{\text{公式提供}} - \underbrace{\sqrt{1-\bar\alpha_s}\,\hat v_\theta}_{\text{網路預測}}$$

$x_s$ 的恆等路徑成為**代數的一部分而非網路的工作**，且係數 $\sqrt{\bar\alpha_s}$ 自帶正確的雜訊層級依賴（$s=1$ 時 0.999、$s=95$ 時 0.062）。

**為何不採可學習的殘差 gate。** 先前曾嘗試 $\hat X = \text{net}(\cdot) + g\cdot x_s$（單一可學習純量 $g$）。由於正確的權重在 $s=1$ 需為 0.999、在 $s=95$ 需為 0.062，單一與 $s$ 無關的純量無法兼顧，訓練中 $g$ 由 0.5 衰減至 0.1，等同無效。v-parameterization 以代數解決了此折衷。

**結果。** 於固定訓練步數的控制實驗與完整 28 期正式訓練中一致：

| 指標 | $X$-prediction | v-parameterization |
|---|---|---|
| 單步：$s=1$ 響應斜率 | 0.0090（最佳的 0.9%） | **0.9981（99.9%）** |
| 全鏈：noise_frac | 0.045 | **0.910** |
| 全鏈：標準差 / 真實 | 0.229 | **0.933** |

振幅問題自此解決，推論時的事後尺度校正不再需要。

### 3.3 發現二：跨資產相關為假性

v-parameterization 修復振幅的同時，**跨資產平均相關由 0.363 崩至 −0.003**（真實 0.396）。追查此現象揭示了一個先前被掩蓋的缺陷。

將每條生成路徑拆解為

$$s_p(t,i) = \underbrace{\mu(t,i)}_{\text{條件均值成分}} + \underbrace{\epsilon_p(t,i)}_{\text{噪聲成分}}, \qquad \mu = \tfrac1P\textstyle\sum_p s_p$$

分別計算兩成分的跨資產相關：

| 模型 | 總相關 | 條件均值成分 | **噪聲成分** | 噪聲變異占比 |
|---|---|---|---|---|
| $X$-prediction | 0.363 | 0.351 | **0.013** | 31% |
| v-parameterization | −0.003 | 0.190 | **−0.020** | 91% |

**兩個模型的噪聲成分相關皆約為 0。** 擴散注入的噪聲 $\varepsilon \sim \mathcal{N}(0,I)$ 在通道間獨立，而 denoiser 從未學會將獨立噪聲映射為相關輸出。因此跨資產相關只能來自條件均值這一個來源：

- $X$-prediction 輸出 69% 為條件均值 $\Rightarrow$ 10 檔共用同一條件訊號 $\Rightarrow$ 相關 0.363
- v-parameterization 輸出 91% 為噪聲 $\Rightarrow$ 相關被稀釋至 0

**結論：原模型看似良好的跨資產結構，是「輸出退化為確定性映射」的副產品，而非模型習得聯合分布。** v-parameterization 並未破壞任何東西，而是使一個一直存在的缺陷顯露。

### 3.4 發現三：條件通道未被有效利用

修復隨機性之後，我們進一步檢驗模型是否使用 condition 中的資訊。

**測試一：條件響應。** 依 condition 自身的波動水準挑出最高與最低的視窗，比較生成輸出的振幅：

| | condition 自身 std | 生成輸出 std |
|---|---|---|
| 低波動視窗 | 0.535 | 0.960 |
| 高波動視窗 | 2.392 | 1.045 |
| **比值** | **4.47×** | **1.07×** |

condition 的波動相差 4.47 倍，輸出僅相差 **1.07 倍** —— 模型幾乎不依 condition 調整輸出振幅。

**測試二：波動聚集。** 以 Ljung–Box $Q(20)$ 檢定 $r^2$ 的自相關（對 i.i.d. 具檢定力）：真實資料 $Q = 455$（10/10 檔拒絕無聚集假設），生成資料 $Q = 43$ —— **波動聚集幾乎完全不存在**。

**理論上這應是可學的。** 在 v-parameterization 下，MSE 的最佳預測為

$$E[X\mid x_s, c] = \frac{\sigma^2_{c}}{\sigma^2_{c} + \text{noise}}\cdot x_s$$

**收縮係數取決於 condition 所指示的條件變異數** —— 高波動 regime 收縮少、低波動 regime 收縮多。因此「依 condition 調整輸出振幅」不僅可學，它根本就是 MSE 的最佳行為。模型未做到，意味著它**無法從 63 日的原始報酬序列中萃取出波動資訊**（波動是二階統計量，在弱訊號、有限樣本下比一階均值難學得多）。

### 3.5 由診斷導出的研究缺口

三項發現共同指向同一結論：

> **TimeDiff 的條件通道目前只傳遞了「條件均值」，未傳遞「條件分布」。** 模型能回答「接下來大概會漲跌多少」，但無法回答「接下來的波動水準、資產間的連動、極端事件的機率會是什麼樣」。

而後者正是金融應用真正需要的資訊，也是 conditional diffusion 相對於單純預測模型的價值所在。這定義了本研究的核心問題：**如何讓條件通道真正承載市場情境的完整資訊，並建立能夠驗證此事的評估方法。**

## 4. 問題形式化

### 4.1 記號

令 $N$ 為資產數、$r_t \in \mathbb{R}^N$ 為第 $t$ 日的對數報酬向量。目標為建模條件分布

$$p_\theta\!\left(X \mid c\right), \qquad X = \left(r_{t+1}, \dots, r_{t+H}\right) \in \mathbb{R}^{H \times N}$$

擴散前向過程為 $x_s = \sqrt{\bar\alpha_s}\,X + \sqrt{1-\bar\alpha_s}\,\varepsilon$，去噪網路採 v-parameterization。

### 4.2 條件的組成

本研究將條件分解為三個語義層次：

$$c = \left[\, c^{\text{hist}},\; c^{\text{fact}},\; c^{\text{scen}} \,\right]$$

| 分量 | 內容 | 性質 |
|---|---|---|
| $c^{\text{hist}}$ | 可觀測歷史：過去 $L$ 日報酬，並**顯式加入已實現波動、橫斷面離散度**等衍生特徵 | 因果、無前視 |
| $c^{\text{fact}}$ | **金融時序知識**：五項 stylized facts 的目標統計量 $\theta^\star = (\alpha_{\text{Hill}},\, \gamma_{\text{skew}},\, \lambda_{\text{jump}},\, \rho_{r^2}(\cdot),\, \ell_{\text{lev}}(\cdot))$，以及跨資產結構 $(\Sigma,\, \Lambda_{\text{PCA}},\, \lambda_L)$ | 由訓練窗估計 |
| $c^{\text{scen}}$ | **目標情境**：欲生成之市場狀態（崩盤、高波動、ETF 成份調整事件等） | 使用者指定或由下游任務決定 |

$c^{\text{hist}}$ 顯式加入波動特徵，直接回應 §3.4 的診斷 —— 與其要求卷積網路自行從原始報酬萃取二階統計量，不如將其作為特徵直接提供。

### 4.3 Progressive Condition

§3.1 的診斷顯示，denoiser 對 $x_s$ 與對 $c$ 的相對依賴隨雜訊層級 $s$ 系統性變化：MSE 最佳響應斜率由 $s$ 小時的 $\bar\alpha \approx 1$ 遞減至 $s$ 大時的 $\approx 0$。這意味著**生成鏈的不同階段在做不同的事**：

- **高雜訊階段（$s$ 大）**：$x_s$ 幾乎不帶資訊，此時決定的是「市場處於何種狀態」——屬於**粗粒度的情境層級**
- **低雜訊階段（$s$ 小）**：$x_s$ 已接近成品，此時決定的是「具體路徑細節」——屬於**細粒度的歷史層級**

據此，條件不應在所有雜訊層級以固定形式注入，而應**隨去噪軌跡漸進釋放**：

$$c(s) = \Pi_s(c)$$

其中 $\Pi_s$ 在高雜訊層級強調 $c^{\text{scen}}$ 與 $c^{\text{fact}}$，在低雜訊層級強調 $c^{\text{hist}}$。此設計使條件的語義粒度與生成鏈各階段的功能對齊。

### 4.4 Guided Noise（Classifier-Free Guidance）

訓練時以機率 $p_{\text{drop}}$ 隨機丟棄條件，使模型同時學得條件與無條件分支。取樣時：

$$\tilde v_\theta(x_s, s, c) = \hat v_\theta(x_s, s, \varnothing) + w \left[\hat v_\theta(x_s, s, c) - \hat v_\theta(x_s, s, \varnothing)\right]$$

引導強度 $w$ 控制「情境依從性」與「多樣性」的權衡。將 $c^{\text{fact}}$ 與 $c^{\text{scen}}$ 編入條件後，$w$ 成為**可調的情境強度旋鈕**：

- $w = 0$：退化為無條件生成
- $w$ 增大：生成路徑更貼合指定的統計特性與市場情境

此機制正是下游壓力測試所需 —— 使用者可指定「接下來發生市場崩盤」並調整 $w$ 以控制情境的極端程度。

### 4.5 待解問題

**P1（條件保真度的可學習性）** — 加入顯式波動特徵與 progressive condition 後，模型能否學得條件異質變異數？驗收標準明確：§3.4 的條件響應比需由 **1.07** 提升至接近 condition 自身的 **4.47**，Ljung–Box $Q$ 需由 43 提升至真實的 455 水準。

**P2（Guided Noise 的有效性）** — 將 $c^{\text{fact}}$、$c^{\text{scen}}$ 編入條件並以 CFG 引導，能否在**不犧牲邊際保真度**的前提下提升情境依從性？$w$ 的最適區間為何？

**P3（評估方法）** — 現有 stylized-facts 指標檢驗的是**無條件邊際分布**，無法量測條件保真度。需要何種指標？

**P4（下游價值）** — 情境條件化的合成資料，能否改善 ETF 成份股變動情境下的投資組合權重轉換策略？

## 5. 評估設計

### 5.1 邊際保真度：兩層指標

所有指標經 $\tau$ 校準轉為 $[0,1]$ 分數：

$$\text{Score}_m = 2^{-D_m/\tau_m}, \qquad \tau_m = \mathbb{E}_b\!\left[D_m\!\left(\hat\theta(X^{(b_1)}_{\text{real}}), \hat\theta(X^{(b_2)}_{\text{real}})\right)\right]$$

$\tau_m$ 為真實資料自身的重抽噪音尺度，以 stationary block bootstrap 估計，**僅使用真實資料**，故不存在 circularity。$D = \tau$ 對應 score 0.5，語義為「誤差恰等於真實估計量自身的抽樣誤差」。

| 層級 | 指標 | 距離定義 | 選擇該距離的理由 |
|---|---|---|---|
| **Level 1**<br>Univariate | Heavy Tail | $\lvert\log(\alpha_{\text{syn}}/\alpha_{\text{real}})\rvert$（左尾 Hill） | $\alpha$ 為正尺度參數，log-ratio 對高估／低估對稱 |
| | Negative Skewness | $\lvert\gamma_{\text{syn}}-\gamma_{\text{real}}\rvert$，符號一致性獨立呈現 | 無量綱且可跨零，比值無定義 |
| | Jump Behaviour | 頻率差 + 幅度對數比 | 門檻由**真實資料**定義並套用於合成，避免 circular evaluation |
| | Volatility Clustering | $\frac1K\sum_k\lvert\rho_{\text{syn}}(k)-\rho_{\text{real}}(k)\rvert$ | 須同時檢驗水平與衰減形狀，單一 lag 不足 |
| | Leverage Effect | $\frac1K\sum_k\lvert\ell_{\text{syn}}(k)-\ell_{\text{real}}(k)\rvert$ | 同上，另加方向診斷 |
| **Level 2**<br>Cross-Asset | Correlation | 上三角 45 對的 MAE | 排除對角線（恆為 1，納入會稀釋誤差） |
| | PCA Structure | $\frac12\sum_k\lvert\text{EVR}_{\text{syn},k}-\text{EVR}_{\text{real},k}\rvert$ | EVR 為機率向量，$\frac12 L_1$ 即 TV distance，天然有界 |
| | Tail Dependence | $\lambda_L(i,j)=P(U_j<u\mid U_i<u)$ 的非對角 MAE | copula 泛函，對邊際單調變換不變 |

其中 $\rho(k)=\text{Corr}(r_t^2,\,r_{t+k}^2)$、$\ell(k)=\text{Corr}(r_t,\,r_{t+k}^2)$，$K = 20 < H = 21$（理由見 §2.5）。

所有 univariate 指標**逐檔計算後平均**（禁止先串接 10 檔，否則會製造 9 個假的跨股票斷點並混合不同尺度的資產），per-asset 結果完整保留供分散度分析。

### 5.2 條件保真度：本研究新增

針對 P3。既有指標的不足在於它們只檢驗無條件邊際分布 —— 一個完全忽略 condition 的模型仍可能在這些指標上得分。本研究提出兩類補充指標：

**(i) 條件響應係數（Conditional Response）。** 對條件的某一維度（如已實現波動）做受控擾動，量測輸出對應統計量的彈性：

$$\text{CR}_j = \frac{\partial \log \widehat{\text{stat}}(X)}{\partial \log c_j}$$

理想值為 1（輸出隨條件等比例變化）。§3.4 的實測對應 $\text{CR} \approx 0.05$，可直接作為改進前的基準線。

**(ii) 情境依從性（Scenario Adherence）。** 給定目標情境 $c^{\text{scen}}$ 及其對應的真實情境樣本集合 $\mathcal{R}(c^{\text{scen}})$：

$$\text{SA}(c^{\text{scen}}) = 1 - \frac{d\!\left(p_\theta(\cdot\mid c^{\text{scen}}),\; \hat p_{\mathcal{R}(c^{\text{scen}})}\right)}{d\!\left(p_\theta(\cdot\mid \varnothing),\; \hat p_{\mathcal{R}(c^{\text{scen}})}\right)}$$

即「條件生成相對於無條件生成，向目標情境靠近了多少」。$\text{SA}=0$ 表示條件毫無作用，$\text{SA}=1$ 表示完全命中目標情境。此指標**必須與 $w$ 一同掃描**，以呈現情境依從性與多樣性的權衡曲線。

### 5.3 評估期間的選擇原則

Primary evaluation window 須在計算任何模型分數**之前**固定，選擇依據僅為真實市場資料的統計性質：

1. **Regime 覆蓋**：需涵蓋平靜期、危機、熊市、復甦
2. **訊噪比**：各項 stylized fact 的真實訊號需高於其自身的重抽噪音 $\tau$
3. **尾部樣本數**：需足以支撐 Hill 估計與尾部相依的計算

依此原則，本研究採 2018-01-02 ~ 2024-12-31（$T=1{,}761$）為主要評估期間；2020（危機）與 2022（熊市）另作 regime robustness 分析，**不參與主要排名**。

## 6. 下游任務：ETF 成份股變動的權重轉換

### 6.1 情境定義

ETF 於再平衡日調整成份股（新增／剔除／權重調整）。追蹤該 ETF 的投資組合須在有限時間內完成權重轉換，面臨三重張力：

- **市場衝擊成本**：轉換速度愈快，滑價愈高
- **追蹤誤差**：轉換期間偏離目標權重
- **情境風險**：若轉換期間遭遇市場崩盤或流動性緊縮，上述兩者同時惡化

令 $w_0$ 為現行權重、$w^\star$ 為目標權重、$\{w_k\}_{k=1}^{K}$ 為轉換路徑，策略須最小化

$$\mathbb{E}_{X \sim p_\theta(\cdot\mid c^{\text{scen}})}\left[\sum_k \underbrace{\mathcal{C}(w_k - w_{k-1})}_{\text{衝擊成本}} + \kappa \underbrace{\text{TE}(w_k; X)}_{\text{追蹤誤差}}\right] + \nu \cdot \text{CVaR}_\beta$$

### 6.2 生成模型的角色

單一歷史軌跡無法提供足夠的情境樣本以最佳化上述目標 —— 特定 ETF 的成份調整事件在歷史上可能只發生過數次。情境條件化生成器可提供：

1. **同一 ETF 調整事件下的多條市場路徑** → 使期望值與 CVaR 可估
2. **指定壓力情境**（$c^{\text{scen}}$ = 崩盤／流動性緊縮，並以 $w$ 調整強度）→ 壓力測試轉換策略
3. **反事實情境** → 評估「若當時市場狀況不同，該策略表現如何」

此三者皆須依賴**條件生成**能力，因而直接對應 §3.5 所界定的研究缺口。

### 6.3 下游評估

- **策略層面**：實現的轉換成本、追蹤誤差、最大回撤
- **風險校準**：以合成資料估計之 VaR / CVaR，在**真實**資料上的實際突破率（coverage backtest）。此為最嚴格的檢驗 —— 若模型低估尾部風險，突破率會顯著高於名目水準。

## 7. 研究問題彙整

| 編號 | 研究問題 | 對應設計 | 驗收標準 |
|---|---|---|---|
| RQ1 | 加入顯式波動特徵與 progressive condition 後，模型能否學得條件異質變異數？ | §4.2–4.3 | CR 由 0.05 → 接近 1；LB $Q$ 由 43 → 455 水準 |
| RQ2 | Progressive condition 與 guided noise 能否在不犧牲邊際保真度下提升情境依從性？ | §4.3–4.4 | SA 隨 $w$ 上升，且 Level 1/2 指標不退步 |
| RQ3 | 如何量化條件保真度，使評估能區分「有無真正使用 condition」？ | §5.2 | CR、SA 對已知失效模型應給出接近 0 的分數 |
| RQ4 | 情境條件化的合成資料能否改善 ETF 權重轉換策略？ | §6 | 轉換成本／追蹤誤差改善；VaR coverage 接近名目水準 |

## 8. 範圍與限制

- **標的與期間**：Dow 10，2018-01-02 ~ 2024-12-31，walk-forward 28 期（後續擴充至 ETF 成份股全集）
- **$c^{\text{fact}}$ 的性質**：由訓練窗估計之結構性先驗，非模型自行習得。論文須明確區分「結構施加」與「模型學習」的貢獻
- **非 free-running simulation**：如 §2.5 所述，生成序列每 21 日重新錨定至真實 condition，故本評估不涵蓋長期自主模擬能力
- **訓練與生成的固有落差**：擴散訓練時 $x_s$ 由真實 $X$ 加噪而得，生成時則自純噪聲出發。實測顯示此落差在低雜訊層級極為顯著（$s$ 小時，訓練期輸出的跨資產相關為 0.481，改以純噪聲輸入則為 −0.001）。**因此任何在訓練期量測的統計量都不能代表生成行為，評估必須在完整生成鏈上進行**
- **統計功效**：$T \approx 252$ 的單年度視窗下，Negative Skewness 與 Leverage Effect 的訊噪比常低於 1，相關結果須標記

---

# English Version

## 1. Background

### 1.1 Motivation: The Single-Trajectory Constraint

Financial applications — risk management, portfolio optimization, and reinforcement-learning trading agents — are fundamentally constrained by the **single realized history**: the market happened only once, and observable extreme events are scarce. For the Dow 10 universe used in this study, across 1,761 trading days (2018–2024), each asset exhibits only about **88** events below its lower 5% tail threshold. Any method requiring tail-risk estimation or a large number of scenario samples is limited by this scarcity.

Generative modeling of financial time series, capable of supplying arbitrarily many market paths, is a natural route past this constraint. Among generative families, **conditional diffusion models** have become a preferred choice, owing to stable training and the ability to steer generation with exogenous information $c$.

### 1.2 Base Model: TimeDiff

This study builds on **TimeDiff** (Shen & Kwok, ICML 2023, *Non-autoregressive Conditional Diffusion Models for Time Series Prediction*), a **non-autoregressive conditional diffusion model** with the following design characteristics:

**(a) Non-autoregressive block generation.** Rather than generating point by point, TimeDiff produces an entire block $X = (r_{t+1},\dots,r_{t+H})$ in one pass, avoiding the error accumulation of autoregressive models. We set $H = 21$ (approximately one trading month).

**(b) Dual-branch conditional encoding.** The conditioning information (the past $L=63$ days of the multivariate series) enters the model through two pathways:

- **Per-variable linear projection**: each variable is independently mapped from its history window to the $H$-day horizon
- **CNN conditioning network**: convolutional encoding across variables, capturing cross-sectional information

Their sum forms a *conditional prior prediction* $\hat{X}_{\text{cond}}$.

**(c) Future Mixup.** During training, this conditional prior is mixed element-wise with the **ground-truth future** $X$ at random proportions:

$$\tilde{X}_{\text{cond}} = m \odot \hat{X}_{\text{cond}} + (1-m) \odot X, \qquad m \sim \mathcal{U}(0,1)^{H\times N}$$

The mechanism resembles scheduled sampling and aims to stabilize conditional alignment. At inference $\tilde{X}_{\text{cond}} = \hat{X}_{\text{cond}}$, as no ground-truth future is available.

**(d) Denoiser input.** The CNN denoiser receives the concatenation

$$\text{input} = \left[\,\underbrace{x_s}_{\text{noised target}},\;\; \underbrace{\tilde{X}_{\text{cond}}}_{\text{conditional prior}},\;\; \underbrace{\text{DLinear}(c^{\text{hist}})}_{\text{short-term extrapolation}}\,\right]$$

together with an embedding of the noise level $s$.

**(e) Sampling.** Reverse generation uses DPM-Solver (probability-flow ODE, second-order multistep) with 20 steps over an $S=100$ discretization under a cosine noise schedule.

TimeDiff suits this study precisely because **conditioning is native to its architecture** — it is exactly the interface into which financial time-series knowledge and target scenarios will be encoded.

### 1.3 Data and Universe

- **Universe**: Dow 10 constituents (AAPL, AMGN, CRM, CSCO, IBM, INTC, MSFT, NKE, VZ, WMT)
- **Features**: three channels per asset — $r^{\text{close}}_t = \log(C_t/C_{t-1})$ (cross-day return), $r^{\text{high}}_t = \log(H_t/C_t)$, $r^{\text{low}}_t = \log(L_t/C_t)$ (intraday range) — for $N \times 3 = 30$ channels in total
- **Stylized facts are defined on the $r^{\text{close}}$ channels only**; intraday-range channels do not enter the statistical evaluation

## 2. Walk-Forward Evaluation Protocol

Evaluation of financial time-series generation is highly susceptible to data leakage. We establish a strict rolling-retraining protocol whose purpose is to **ensure that generation at any point in time uses only information available before that point**.

### 2.1 Temporal Partitioning

| Parameter | Value | Note |
|---|---|---|
| Generation period | 2018-01-02 to 2024-12-31 | 1,761 common trading days |
| Retraining stride | 63 trading days | approximately one quarter |
| Number of cycles | 28 | final cycle covers 60 days |
| Training window length | 1,260 trading days | approximately five years |
| Condition length $L$ | 63 days | approximately one quarter |
| Generation block $H$ | 21 days | approximately one month |

The origin of cycle $k$ is the $63k$-th trading day of the generation period.

### 2.2 Two-Stage Training per Cycle

**Stage A — model selection.** The first 1,008 days of the training window are used for training and the last 252 for validation; the scaler is fit on the first 1,008 days only. Validation MSE is evaluated at fixed step intervals and the best step $k^\star$ is selected.

**Stage B — final training.** The model and scaler are **re-initialized from scratch** (no warm start) and trained on the full 1,260 days up to step $k^\star$.

Both stages advance in optimizer steps rather than epochs, since the two stages have different numbers of steps per epoch (925 versus 1,177 training windows). Learning-rate schedules and loss annealing are likewise indexed by step, ensuring that "the same step" carries identical hyperparameter state in both stages — without which the selected $k^\star$ would be meaningless.

### 2.3 Look-Ahead Guarantees

Four mechanisms prevent leakage:

1. **Training window strictly precedes the origin**: data are taken from $[\text{origin}-1260,\ \text{origin})$, enforced in code by an assertion that the last training day $<$ origin
2. **The scaler is fit on the training segment only**; validation and generation periods are transformed with that same scaler
3. **Conditions use only the real history available at that time**: for block $j$ of cycle $k$, the condition is $\text{real}[\,\text{origin}+21j-63 : \text{origin}+21j\,]$
4. **Generated data are never fed back into training**: every cycle trains on real data only

### 2.4 Generation and Price Reconstruction

Each cycle generates $63 = 21 \times 3$ days in three blocks, with each block conditioned on the real 63 days available at that moment.

After concatenating returns across all 1,761 days, prices are reconstructed recursively from a **single seed**:

$$C_t = C_{t-1}\exp\!\left(r^{\text{close}}_t\right), \qquad C_0 = \text{last real close of 2017}$$

**No reset occurs at block or retraining boundaries** — the price on day $t$ chains onto the *generated* price of day $t-1$, never onto the real price. This ensures the price trajectory is entirely a product of the model.

### 2.5 Protocol Limitation (to be disclosed explicitly)

The generated series is **not a free-running simulation**: every 21 trading days it is re-anchored to a real 63-day condition. Consequently:

- Temporal dependence beyond 21 lags derives in part from the real condition rather than the model
- The lag ceiling for all temporal metrics (volatility clustering, leverage effect) must be $K = 20 < H = 21$, otherwise the measurement reflects a protocol artefact
- This evaluation **cannot** speak to the model's long-horizon autonomous simulation capability

## 3. Preliminary Empirical Diagnostics

We diagnosed TimeDiff systematically under the above protocol, yielding findings that define this study's point of departure.

### 3.1 Finding I: The Generative Process Degenerates into a Deterministic Map

**Symptom.** Generated return standard deviation is **22.9%** of real; exceedances below the real 5% lower-tail threshold fall from 88 to **0.32**. The model virtually never produces extreme events of realistic magnitude.

**Confirmation.** Holding the condition fixed and generating once for each of 32 independent diffusion noises, we decompose the output variance as

$$\text{noise\_frac} = \frac{\mathrm{Var}_k(s)}{\mathrm{Var}_k(s) + \mathrm{Var}_{t,i}(\mu)}, \qquad \mu = \tfrac1K\textstyle\sum_k s_k$$

Measured **noise_frac $= 0.045$** — only 4.5% of output variance originates from the diffusion noise; the remaining 95.5% is determined by the condition. The model is effectively a deterministic regressor.

**Root cause.** At low noise levels, $x_s = \sqrt{\bar\alpha_s}X + \sqrt{1-\bar\alpha_s}\varepsilon \approx X$, and the MSE-optimal prediction is $E[X\mid x_s] = \sqrt{\bar\alpha_s}\,x_s$, whose regression coefficient on $X$ equals $\bar\alpha_s$. Measured denoiser response slopes:

| $s$ | $\bar\alpha_s$ (optimal slope) | measured slope | measured/optimal |
|---|---|---|---|
| 1 | 0.9983 | 0.0090 | **0.90%** |
| 10 | 0.9667 | 0.0092 | 0.95% |
| 40 | 0.6325 | 0.0074 | 1.17% |
| 95 | 0.0039 | 0.0004 | 10.3% |

At $s=1$ — where the input is 99.8% original data and the answer is essentially written on the input — the network responds at only **0.9%** of the required level. Substituting a perfect predictor $\sqrt{\bar\alpha}x_s$ into the identical measurement procedure recovers the theoretical value (0.9979 versus 0.9983), confirming the measurement itself is unbiased.

**Causal chain.** The sole entry point for diffusion noise is $x_s$. The network does not use $x_s$ $\Rightarrow$ noise information cannot propagate $\Rightarrow$ output degenerates into a function of the condition $\Rightarrow$ noise_frac collapses. **The single-step slope failure is the cause; the full-chain collapse is the effect.**

### 3.2 First Correction: v-parameterization

**Nature of the problem.** Under $X$-prediction (the original setting), the network must **learn the identity map through convolutional layers** in order to convey $x_s$ to the output. On near-unpredictable return series, "emit a smooth conditional mean" is a locally attractive shortcut under MSE, and the network takes it.

**Solution.** Adopt v-parameterization (Salimans & Ho, 2022):

$$v = \sqrt{\bar\alpha_s}\,\varepsilon - \sqrt{1-\bar\alpha_s}\,X, \qquad \hat{X} = \underbrace{\sqrt{\bar\alpha_s}\,x_s}_{\text{supplied by the formula}} - \underbrace{\sqrt{1-\bar\alpha_s}\,\hat v_\theta}_{\text{predicted by the network}}$$

The identity path for $x_s$ becomes **part of the algebra rather than the network's job**, and the coefficient $\sqrt{\bar\alpha_s}$ carries the correct noise-level dependence intrinsically (0.999 at $s=1$; 0.062 at $s=95$).

**Why not a learnable residual gate.** We previously attempted $\hat X = \text{net}(\cdot) + g\cdot x_s$ with a single learnable scalar $g$. Since the correct weight must be 0.999 at $s=1$ and 0.062 at $s=95$, a single $s$-independent scalar cannot serve both ends; during training $g$ decayed from 0.5 to 0.1, rendering the mechanism inert. v-parameterization resolves this trade-off algebraically.

**Results**, consistent across a fixed-step control experiment and the full 28-cycle training:

| Metric | $X$-prediction | v-parameterization |
|---|---|---|
| Single-step slope at $s=1$ | 0.0090 (0.9% of optimal) | **0.9981 (99.9%)** |
| Full-chain noise_frac | 0.045 | **0.910** |
| Full-chain std / real | 0.229 | **0.933** |

The amplitude problem is thereby resolved, and post-hoc scale correction at inference is no longer required.

### 3.3 Finding II: Cross-Asset Correlation Is Spurious

While v-parameterization restored amplitude, **mean cross-asset correlation collapsed from 0.363 to −0.003** (real: 0.396). Investigating this revealed a previously masked deficiency.

Decomposing each generated path as

$$s_p(t,i) = \underbrace{\mu(t,i)}_{\text{conditional-mean component}} + \underbrace{\epsilon_p(t,i)}_{\text{noise component}}, \qquad \mu = \tfrac1P\textstyle\sum_p s_p$$

and computing cross-asset correlation for each component:

| Model | Total | Conditional-mean component | **Noise component** | Noise variance share |
|---|---|---|---|---|
| $X$-prediction | 0.363 | 0.351 | **0.013** | 31% |
| v-parameterization | −0.003 | 0.190 | **−0.020** | 91% |

**In both models the noise component carries essentially zero cross-asset correlation.** The injected diffusion noise $\varepsilon \sim \mathcal{N}(0,I)$ is independent across channels, and the denoiser never learned to map independent noise to correlated output. Cross-asset correlation can therefore originate only from the conditional mean:

- $X$-prediction: 69% of output is conditional mean $\Rightarrow$ ten assets share one conditional signal $\Rightarrow$ correlation 0.363
- v-parameterization: 91% of output is noise $\Rightarrow$ correlation diluted to zero

**Conclusion: the original model's apparently adequate cross-asset structure is a by-product of output degeneracy, not evidence of a learned joint distribution.** v-parameterization broke nothing; it exposed a defect that had been present throughout.

### 3.4 Finding III: The Conditioning Channel Is Underused

Having restored stochasticity, we further examined whether the model exploits the information in the condition.

**Test 1: conditional response.** Selecting the highest- and lowest-volatility conditioning windows and comparing generated output amplitude:

| | condition std | output std |
|---|---|---|
| Low-volatility windows | 0.535 | 0.960 |
| High-volatility windows | 2.392 | 1.045 |
| **Ratio** | **4.47×** | **1.07×** |

A 4.47-fold difference in conditioning volatility produces only a **1.07-fold** difference in output — the model barely modulates amplitude according to the condition.

**Test 2: volatility clustering.** Ljung–Box $Q(20)$ on $r^2$ (which has power against i.i.d. series): real data $Q = 455$ (rejecting the no-clustering hypothesis for 10/10 assets); generated data $Q = 43$ — **volatility clustering is essentially absent**.

**This should in principle be learnable.** Under v-parameterization, the MSE-optimal prediction is

$$E[X\mid x_s, c] = \frac{\sigma^2_{c}}{\sigma^2_{c} + \text{noise}}\cdot x_s$$

where **the shrinkage factor depends on the conditional variance implied by $c$** — less shrinkage in high-volatility regimes, more in low-volatility ones. Modulating output amplitude by the condition is therefore not merely learnable; it *is* the MSE-optimal behaviour. The model's failure to do so indicates it **cannot extract volatility information from 63 days of raw returns** — volatility being a second-order statistic, materially harder to learn than the first-order mean under weak signal and limited samples.

### 3.5 The Research Gap Implied by the Diagnostics

The findings converge on a single conclusion:

> **TimeDiff's conditioning channel currently conveys only the conditional mean, not the conditional distribution.** The model can answer "roughly how much will prices move next," but not "what will the volatility level, the co-movement across assets, and the probability of extreme events look like."

The latter is precisely what financial applications require, and precisely where conditional diffusion should add value over a plain forecasting model. This defines the core problem of this study: **how to make the conditioning channel genuinely carry the full information of a market scenario, and how to build an evaluation methodology capable of verifying that it does.**

## 4. Problem Formalization

### 4.1 Notation

Let $N$ be the number of assets and $r_t \in \mathbb{R}^N$ the log-return vector on day $t$. The objective is to model

$$p_\theta\!\left(X \mid c\right), \qquad X = \left(r_{t+1}, \dots, r_{t+H}\right) \in \mathbb{R}^{H \times N}$$

with forward process $x_s = \sqrt{\bar\alpha_s}\,X + \sqrt{1-\bar\alpha_s}\,\varepsilon$ and a v-parameterized denoiser.

### 4.2 Composition of the Condition

$$c = \left[\, c^{\text{hist}},\; c^{\text{fact}},\; c^{\text{scen}} \,\right]$$

| Component | Content | Nature |
|---|---|---|
| $c^{\text{hist}}$ | Observable history: past $L$ days of returns, **augmented with explicit realized-volatility and cross-sectional-dispersion features** | Causal, look-ahead-free |
| $c^{\text{fact}}$ | **Financial time-series knowledge**: target statistics of the five stylized facts $\theta^\star = (\alpha_{\text{Hill}},\, \gamma_{\text{skew}},\, \lambda_{\text{jump}},\, \rho_{r^2}(\cdot),\, \ell_{\text{lev}}(\cdot))$ and cross-asset structure $(\Sigma,\, \Lambda_{\text{PCA}},\, \lambda_L)$ | Estimated from the training window |
| $c^{\text{scen}}$ | **Target scenario**: the market state to be generated (crash, high volatility, ETF rebalancing event) | User-specified or set by the downstream task |

The explicit volatility features in $c^{\text{hist}}$ respond directly to the diagnostic in §3.4: rather than requiring the convolutional network to extract second-order statistics from raw returns, we supply them as features.

### 4.3 Progressive Condition

The diagnostic in §3.1 shows that the denoiser's relative reliance on $x_s$ versus $c$ varies systematically with the noise level $s$: the MSE-optimal response slope declines from $\bar\alpha \approx 1$ at small $s$ to $\approx 0$ at large $s$. This implies that **different stages of the generation chain perform different functions**:

- **High-noise stage (large $s$)**: $x_s$ carries almost no information; what is decided here is *which market state we are in* — a **coarse, scenario-level** determination
- **Low-noise stage (small $s$)**: $x_s$ is already close to the final output; what is decided here is *the specific path details* — a **fine, history-level** determination

Accordingly, the condition should not be injected in a fixed form at all noise levels, but **released progressively along the denoising trajectory**:

$$c(s) = \Pi_s(c)$$

where $\Pi_s$ emphasizes $c^{\text{scen}}$ and $c^{\text{fact}}$ at high noise levels and $c^{\text{hist}}$ at low noise levels, aligning the semantic granularity of the condition with the function of each stage.

### 4.4 Guided Noise (Classifier-Free Guidance)

The condition is dropped with probability $p_{\text{drop}}$ during training so that the model learns both conditional and unconditional branches. At sampling:

$$\tilde v_\theta(x_s, s, c) = \hat v_\theta(x_s, s, \varnothing) + w \left[\hat v_\theta(x_s, s, c) - \hat v_\theta(x_s, s, \varnothing)\right]$$

The guidance scale $w$ governs the trade-off between scenario adherence and diversity. With $c^{\text{fact}}$ and $c^{\text{scen}}$ encoded into $c$, $w$ becomes a **tunable scenario-intensity control**:

- $w = 0$: reduces to unconditional generation
- Larger $w$: paths conform more closely to the specified statistical properties and market scenario

This is exactly the mechanism required for downstream stress testing: the user specifies "a crash follows" and tunes $w$ to control the severity of the scenario.

### 4.5 Problems to Be Addressed

**P1 (Learnability of conditional fidelity)** — With explicit volatility features and progressive conditioning, can the model learn conditional heteroscedasticity? The acceptance criterion is explicit: the conditional response ratio of §3.4 must rise from **1.07** toward the conditioning ratio of **4.47**, and Ljung–Box $Q$ from 43 toward the real level of 455.

**P2 (Effectiveness of guided noise)** — Does encoding $c^{\text{fact}}$ and $c^{\text{scen}}$ and applying CFG improve scenario adherence **without sacrificing marginal fidelity**? What is the appropriate range for $w$?

**P3 (Evaluation methodology)** — Existing stylized-facts metrics examine *unconditional marginal* distributions and cannot measure conditional fidelity. What metrics are required?

**P4 (Downstream value)** — Does scenario-conditioned synthetic data improve portfolio weight-transition strategies under ETF constituent changes?

## 5. Evaluation Design

### 5.1 Marginal Fidelity: Two-Level Metrics

All metrics are $\tau$-calibrated into $[0,1]$ scores:

$$\text{Score}_m = 2^{-D_m/\tau_m}, \qquad \tau_m = \mathbb{E}_b\!\left[D_m\!\left(\hat\theta(X^{(b_1)}_{\text{real}}), \hat\theta(X^{(b_2)}_{\text{real}})\right)\right]$$

$\tau_m$ is the resampling noise floor of the real data itself, estimated by stationary block bootstrap **using real data only**, so no circularity arises. $D = \tau$ maps to a score of 0.5, meaning "the error equals the sampling error of the real estimator itself."

| Level | Metric | Distance | Rationale for that distance |
|---|---|---|---|
| **Level 1**<br>Univariate | Heavy Tail | $\lvert\log(\alpha_{\text{syn}}/\alpha_{\text{real}})\rvert$ (left-tail Hill) | $\alpha$ is a positive scale parameter; log-ratio treats over- and under-estimation symmetrically |
| | Negative Skewness | $\lvert\gamma_{\text{syn}}-\gamma_{\text{real}}\rvert$, sign agreement reported separately | Dimensionless and sign-crossing; ratios undefined |
| | Jump Behaviour | Frequency difference + log-ratio of magnitude | Threshold defined on **real data** and applied to synthetic, avoiding circular evaluation |
| | Volatility Clustering | $\frac1K\sum_k\lvert\rho_{\text{syn}}(k)-\rho_{\text{real}}(k)\rvert$ | Must test both level and decay shape; a single lag is insufficient |
| | Leverage Effect | $\frac1K\sum_k\lvert\ell_{\text{syn}}(k)-\ell_{\text{real}}(k)\rvert$ | As above, plus a directional diagnostic |
| **Level 2**<br>Cross-Asset | Correlation | MAE over the 45 upper-triangular pairs | Diagonal excluded (identically 1; inclusion dilutes the error) |
| | PCA Structure | $\frac12\sum_k\lvert\text{EVR}_{\text{syn},k}-\text{EVR}_{\text{real},k}\rvert$ | EVR is a probability vector; $\frac12 L_1$ is total variation, naturally bounded |
| | Tail Dependence | Off-diagonal MAE of $\lambda_L(i,j)=P(U_j<u\mid U_i<u)$ | A copula functional, invariant to monotone marginal transforms |

with $\rho(k)=\text{Corr}(r_t^2,\,r_{t+k}^2)$, $\ell(k)=\text{Corr}(r_t,\,r_{t+k}^2)$, and $K = 20 < H = 21$ (see §2.5).

All univariate metrics are computed **per asset and then averaged** (concatenating the ten assets is prohibited, as it would create nine spurious cross-asset breakpoints and blend assets of differing scale); per-asset results are retained for dispersion analysis.

### 5.2 Conditional Fidelity: Contribution of This Study

Addressing P3. Existing metrics examine only unconditional marginals — a model that entirely ignores the condition may still score well on them. We propose two complementary measures:

**(i) Conditional Response.** Applying a controlled perturbation to one dimension of the condition (e.g. realized volatility), we measure the elasticity of the corresponding output statistic:

$$\text{CR}_j = \frac{\partial \log \widehat{\text{stat}}(X)}{\partial \log c_j}$$

with an ideal value of 1. The measurement in §3.4 corresponds to $\text{CR} \approx 0.05$, serving directly as the pre-improvement baseline.

**(ii) Scenario Adherence.** Given a target scenario $c^{\text{scen}}$ and its matching set of real episodes $\mathcal{R}(c^{\text{scen}})$:

$$\text{SA}(c^{\text{scen}}) = 1 - \frac{d\!\left(p_\theta(\cdot\mid c^{\text{scen}}),\; \hat p_{\mathcal{R}(c^{\text{scen}})}\right)}{d\!\left(p_\theta(\cdot\mid \varnothing),\; \hat p_{\mathcal{R}(c^{\text{scen}})}\right)}$$

that is, how much closer conditional generation moves toward the target scenario relative to unconditional generation. $\text{SA}=0$ indicates the condition has no effect; $\text{SA}=1$ indicates the target scenario is fully matched. This metric **must be swept jointly with $w$** to reveal the adherence–diversity trade-off curve.

### 5.3 Principles for Selecting the Evaluation Period

The primary evaluation window must be fixed **before** any model score is computed, on the basis of properties of the real data alone:

1. **Regime coverage**: must span calm periods, crisis, bear market, and recovery
2. **Signal-to-noise ratio**: the real signal of each stylized fact must exceed its own resampling noise floor $\tau$
3. **Tail sample size**: sufficient to support Hill estimation and tail-dependence computation

On this basis we adopt 2018-01-02 to 2024-12-31 ($T=1{,}761$) as the primary window; 2020 (crisis) and 2022 (bear market) serve as regime-robustness analyses and **do not participate in the primary ranking**.

## 6. Downstream Task: Weight Transition under ETF Constituent Changes

### 6.1 Scenario Definition

An ETF adjusts its constituents on rebalancing dates (additions, deletions, weight changes). A tracking portfolio must complete the corresponding weight transition within a limited window, facing three competing pressures:

- **Market impact cost**: faster transition incurs greater slippage
- **Tracking error**: deviation from target weights during the transition
- **Scenario risk**: a crash or liquidity squeeze during the transition worsens both simultaneously

With $w_0$ the current weights, $w^\star$ the target, and $\{w_k\}_{k=1}^{K}$ the transition path, the strategy minimizes

$$\mathbb{E}_{X \sim p_\theta(\cdot\mid c^{\text{scen}})}\left[\sum_k \underbrace{\mathcal{C}(w_k - w_{k-1})}_{\text{impact cost}} + \kappa \underbrace{\text{TE}(w_k; X)}_{\text{tracking error}}\right] + \nu \cdot \text{CVaR}_\beta$$

### 6.2 Role of the Generative Model

A single historical trajectory cannot supply enough scenario samples to optimize this objective — a given ETF's constituent changes may have occurred only a handful of times historically. A scenario-conditioned generator provides:

1. **Many market paths under the same rebalancing event** → expectations and CVaR become estimable
2. **Specified stress scenarios** ($c^{\text{scen}}$ = crash / liquidity squeeze, with severity tuned via $w$) → stress-testing of the transition policy
3. **Counterfactual scenarios** → assessment of how the policy would have fared under different market conditions

All three depend on genuine **conditional** generation, corresponding directly to the research gap identified in §3.5.

### 6.3 Downstream Evaluation

- **Strategy level**: realized transition cost, tracking error, maximum drawdown
- **Risk calibration**: VaR / CVaR estimated on synthetic data, assessed by realized exceedance rates on **real** data (coverage backtest). This is the most stringent test — a model that understates tail risk will produce exceedance rates significantly above the nominal level.

## 7. Summary of Research Questions

| No. | Research Question | Design | Acceptance Criterion |
|---|---|---|---|
| RQ1 | With explicit volatility features and progressive conditioning, can the model learn conditional heteroscedasticity? | §4.2–4.3 | CR from 0.05 → near 1; LB $Q$ from 43 → 455 level |
| RQ2 | Do progressive conditioning and guided noise improve scenario adherence without sacrificing marginal fidelity? | §4.3–4.4 | SA increases with $w$ while Level 1/2 metrics do not regress |
| RQ3 | How can conditional fidelity be quantified so evaluation distinguishes genuine conditioning from none? | §5.2 | CR and SA must score near zero for models known to be inert |
| RQ4 | Does scenario-conditioned synthetic data improve ETF weight-transition strategies? | §6 | Improved transition cost / tracking error; VaR coverage near nominal |

## 8. Scope and Limitations

- **Universe and period**: Dow 10, 2018-01-02 to 2024-12-31, 28 walk-forward cycles (to be extended to the full ETF constituent set)
- **Nature of $c^{\text{fact}}$**: a structural prior estimated from the training window, not learned by the model. The thesis must explicitly separate the contribution of *structural imposition* from that of *model learning*
- **Not a free-running simulation**: as noted in §2.5, the generated series is re-anchored to a real condition every 21 days; this evaluation therefore does not cover long-horizon autonomous simulation
- **Intrinsic train–generate gap**: during diffusion training $x_s$ is obtained by noising the real $X$, whereas generation begins from pure noise. Measurements show this gap is severe at low noise levels (at small $s$, training-time cross-asset correlation of the output is 0.481 versus −0.001 with pure-noise input). **Consequently no statistic measured during training can represent generative behaviour; evaluation must be conducted on the full generation chain**
- **Statistical power**: under single-year windows of $T \approx 252$, the signal-to-noise ratios of Negative Skewness and Leverage Effect frequently fall below 1, and such results must be flagged
