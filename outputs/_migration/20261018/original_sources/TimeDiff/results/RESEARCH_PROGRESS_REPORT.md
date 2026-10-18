# Synthetic Financial Market Generation — 研究進度報告 0823

**對象**：Dow 10 股票的 multivariate synthetic financial time series
**評估期間**：2018-01-02 ~ 2024-12-31（1,761 個交易日，28 期 walk-forward 滾動重訓）
**基礎模型**：TimeDiff（conditional diffusion，CNN U-Net denoiser + DPM-Solver）

---

## 摘要

本階段從一個關鍵診斷出發：原始模型生成的報酬**變異數只有真實資料的 21%**。追查後發現這不是超參數問題，而是**denoiser 的 x_t 路徑失效**（低雜訊時的斜率只有 MSE 最佳解的 2.5%）。

修正這個問題（改用 v-parameterization）後，振幅回到 93%，但**跨資產相關從 0.363 崩到 −0.003**。進一步的變異數分解顯示：所有模型的**噪聲成分相關皆 ≈ 0**——denoiser 從未學會把獨立噪聲映射成相關輸出；原始模型的高相關只是「輸出退化成確定性條件均值」的副產品。

由此建立了一個**四層結構化生成框架**：把模型學不到的統計性質改由結構保證，模型只負責條件動態。最終配置在主要評估期間取得 **Overall 0.476**，為所有模型（含兩個 null model）之冠。

**同時必須指出的限制**：一個不含任何生成模型的多元高斯（Gaussian(Σ̂)）取得 Overall 0.446，與最佳模型僅差 0.030。diffusion 模型的可證明貢獻集中在**跨資產結構**，stylized facts 的改善主要來自結構層而非模型。

---

## 1. 起點與問題定位

### 1.1 原始狀態

| 指標 | 原始模型 (x_start) | 真實資料 |
|---|---|---|
| VolRatio（標準差比） | **0.229** | 1.000 |
| Hill α（尾部指數） | 3.862 | 2.610 |
| 真實 q95 門檻下的跳躍曝險 | **0.32 筆** | 88 筆 |
| 跨資產平均相關 | 0.363 | 0.396 |

模型幾乎不產生真實幅度的極端事件。先前的紀錄把此歸因於「x0-MSE 的固有限制」，並以推論時的 `vol_scale=3.86` 事後校正振幅。

### 1.2 診斷一：denoiser 的 x_t 路徑失效

擴散模型在低雜訊步（t=1，$\bar\alpha=0.998$）時，$x_t \approx x_0$，**MSE 的最佳解就是把 $x_t$ 複製到輸出**，斜率應為 $\sqrt{\bar\alpha}\approx 1$。

實測結果：

| t | $\bar\alpha$ | MSE 最佳斜率 $\sqrt{\bar\alpha}$ | 實測斜率 | 實測/最佳 |
|---|---|---|---|---|
| 1 | 0.9983 | 0.999 | 0.0090 | **0.9%** |
| 10 | 0.9667 | 0.983 | 0.0092 | 0.9% |
| 40 | 0.6325 | 0.795 | 0.0074 | 0.9% |
| 95 | 0.0039 | 0.062 | 0.0004 | 0.6% |

**關鍵觀察**：實測/最佳的比值在所有 t 都約等於 0.009。denoiser 學到了正確的 **t 依賴形狀**，但整條 $x_t$ 路徑被衰減約 100 倍。

**結論**：「x0-MSE 必然導致收縮」只在高 t 成立。低 t 的最佳解幾乎是恆等映射，而模型根本沒學到——這是**架構問題**（缺少 $x_t \to$ 輸出的恆等路徑），不是目標函數的宿命。

這也解釋了先前 `use_xt_residual`（可學習純量 gate）失敗的原因：單一與 t 無關的 gate 無法同時滿足 t=1 需要 1.0、t=95 需要 0.06，訓練只能收斂到一個很小的折衷值。

### 1.3 診斷二：condition collapse

以 stationary block bootstrap 對同一 condition 抽 32 個樣本，做變異數分解：

$$\text{noise\_frac} = \frac{\mathrm{Var}_k(s)}{\mathrm{Var}_k(s) + \mathrm{Var}_{t,i}(\mu)}$$

| Cycle | noise_frac | 標準化空間樣本 std |
|---|---|---|
| 13 (2021-04) | **0.039** | 0.131 |
| 27 (2024-10) | **0.056** | 0.107 |

**只有 4–6% 的輸出變異來自擴散噪聲**，其餘 94–96% 由 condition 決定。模型實質上是確定性回歸器。

> **方法註記**：初版的 noise_frac 計算取跨 condition 的平均，被少數「取樣爆衝」樣本主導（cycle 0 的 8 個 condition 中有 4 個 var_noise 高出百倍）。改為逐 condition 計算取中位數後才得到可靠數值。同時 seed_corr 必須扣除條件均值再計算，否則會被「不同 ticker 的均值本來就不同」灌水到 0.96。

---

## 2. 第一層修正：v-parameterization

### 2.1 動機與機制

$x_0$-prediction 下，網路必須**從卷積層自己學出恆等映射**。v-parameterization（Salimans & Ho 2022）的還原式為

$$v = \sqrt{\bar\alpha}\,\varepsilon - \sqrt{1-\bar\alpha}\,x_0, \qquad x_0 = \sqrt{\bar\alpha}\,x_t - \sqrt{1-\bar\alpha}\,\hat v$$

$x_t$ 的恆等路徑是**代數上直接寫死的**，且係數 $\sqrt{\bar\alpha}$ 自帶正確的 t 依賴。網路只需學殘差項。

### 2.2 為什麼不是 ε-parameterization

ε 同樣內建恆等路徑，但還原式 $x_0=(x_t-\sqrt{1-\bar\alpha}\,\hat\varepsilon)/\sqrt{\bar\alpha}$ 的分母在高 t（$\bar\alpha\to0$）趨近零，誤差被放大。實測 ε 版的標準化樣本 std 達 **1687.8**（目標 1.0）——完全發散。v 的兩個係數都有界，數值穩定。

> **先前結論的修正**：舊紀錄稱「ε-parameterization 爆炸」，我原本推測那是被迫使用 ancestral sampling 造成的 confound。本次以 DPM-Solver 重測，ε 仍然發散（std 1689）——**ε 本身在此設定下就不穩定，與取樣器無關**。

### 2.3 實作範圍

底層 `dpm_solver.py` 原生支援 `model_type ∈ {noise, x_start, v}`（第 346 行），但上層有兩處人為限制：

| 位置 | 原本 | 改為 |
|---|---|---|
| `DDPM.py:48` | `assert parameterization == "x_start"` | `in ("x_start","noise","v")` |
| `dpm_sampler.py:76` | `model_type="x_start"` 寫死 | `model_type=self.parameterization` |
| `dpm_sampler.py:84` | `predict_x0=True` | `predict_x0=(param=="x_start")` |

**v 的訓練路徑為新增程式碼**（`get_v()`、`predict_start_from_v()`、target 分支、`lvlb_weights` 分支）——git HEAD 版本的 `diffusion_worker.py:41` 明確只支援 `["noise","x_start"]`。

### 2.4 結果

固定訓練步數（550 步）的控制實驗，排除「訓練長度不同」的干擾：

| 參數化 | noise_frac | 標準化 sample std | 低 t 斜率 | 斜率/最佳 |
|---|---|---|---|---|
| x_start | 0.008 | 0.320 | 0.026 | **2.6%** |
| noise | 0.962 | **1689.4** ⚠️ | 0.987 | 99.7% |
| **v** | **0.905** | **0.984** | **0.983** | **99.3%** |

VolRatio 從 0.229 提升到 **0.933**，`vol_scale` 的事後校正不再需要。

---

## 3. 第二層修正前的關鍵發現：跨資產相關是假性的

v-parameterization 使 Stylized Facts 從 0.238 提升到 0.321，但 **Cross-Asset 從 0.375 崩到 0.026**。

### 3.1 變異數分解

把每條路徑拆成「條件均值成分」（跨 50 條路徑取平均）與「噪聲成分」（殘差）：

| Model | 總相關 | 條件均值相關 | **噪聲成分相關**（逐年中位數） | 噪聲變異占比 |
|---|---|---|---|---|
| x_start baseline | 0.363 | 0.351 | **0.013** | 31.3% |
| v-prediction | −0.003 | 0.190 | **−0.020** | 91.2% |
| v + Cross Loss | 0.141 | 0.666 | 0.009 | 80.0% |

**四個模型的噪聲成分相關全部 ≈ 0。** 擴散注入的噪聲 $\varepsilon\sim N(0,I)$ 在通道間獨立，denoiser 從未學會把獨立噪聲映射成相關輸出。

因此：
- baseline 輸出 69% 是條件均值 → 10 檔共用同一條件訊號 → 相關 0.363
- v-prediction 輸出 91% 是噪聲 → 相關被稀釋到 0

**baseline 的高 Cross-Asset 分數是「噪聲成分少」的副產品，不是學到了聯合分布。**

> **方法註記**：全期計算時 baseline 的噪聲相關為 0.432，逐年拆解後發現只有 2018–2020 偏高，且其殘差變異的 **42.4%** 集中在最極端 1% 的交易日——是早期 cycle 的取樣爆衝所致。v-prediction 的對應數字僅 3.0%。

### 3.2 Cross-Asset Loss 的嘗試與失敗

在總損失加入 $L_{\text{cross}} = \|\mathrm{Corr}(r_{gen}) - \mathrm{Corr}(r_{real})\|_F^2$（10×10 資產相關矩陣，以 batch×time 為觀測值，不 flatten 資產維度）。

單元測試證實 loss 正確（獨立資料 $L_F^2$=15.6、完美匹配 0.18、梯度只流向 rclose 通道）。但**訓練時量到的 corr_mae 只有 0.035**，而生成序列是 0.40——loss 幾乎沒有梯度壓力。

追查後發現**三層資訊洩漏**，使訓練時的統計量無法代表生成行為：

| 障礙 | 機制 | 量化 |
|---|---|---|
| A | denoiser 的 $x_t$ 路徑失效 | 斜率 2.5% → v-prediction 已解決 |
| B | **future mixup**：`DDPM_CNNNet.py:291` 將真實 $x_0$ 以 $U(0,1)$ 隨機比例混入 denoiser 輸入 | t=99 時訓練相關 0.61 / 推論 0.19 |
| C | $x_t=\sqrt{\bar\alpha}x_0+\sqrt{1-\bar\alpha}\varepsilon$ 永遠殘留真實 $x_0$ | t=95 時仍有 6% 殘留 |

障礙 B 是 TimeDiff 原論文的設計（future mixup），對**點預測**合理（類似 scheduled sampling），但對**生成**任務造成 exposure bias。

針對 B 與 C 各實作了修正（`--aux_no_teacher_forcing`、`--cross_gen_aligned`），generation-aligned 模式使訓練時的 corr_mae 從 0.035 升到 **0.266**（7.6 倍），確實看到了真正的誤差。但完整訓練後：

| 配置 | SF Score | CA Score | Overall |
|---|---|---|---|
| v-prediction | 0.321 | 0.026 | 0.173 |
| v + Cross Loss | 0.324 | 0.073 | 0.198 |
| v + High-Noise Cross Loss | 0.243 | 0.100 | 0.172 |

**改善幅度遠不足**（CA 從 0.026 到最多 0.100，目標水準是 0.5+），且 High-Noise 版以 SF −0.078 換取 CA +0.075，是不利的交換（它靠放大振幅湊相關，VolRatio 衝到 1.602）。

**結論**：讓 loss「量對對象」不等於「模型能無代價地滿足它」。以 loss 教模型學習跨資產相依，在此架構下效率過低。

---

## 4. 四層結構化生成框架

由上述失敗歸納出設計原則：**把模型學不到的統計性質改由結構保證，模型只負責它做得到的部分（給定 condition 生成標準化殘差）。**

### 4.1 第二層：Cholesky 白化（跨資產相關）

令訓練窗的共變異 $\Sigma = LL^\top$：

$$z = r(L^{-1})^\top \Rightarrow \mathrm{cov}(z)=I, \qquad r = zL^\top$$

模型在 $z$ 空間訓練與生成。因為 $L\cdot N(0,I)=N(0,\Sigma)$，這與「相關噪聲先驗」**數學上等價**，但在 $z$ 空間噪聲仍是各向同性，**DDPM 的 posterior 係數與 DPM-Solver 的 probability-flow ODE 推導都不需要修改**。

**為什麼不直接改 `_draw_noise` 為 `L @ randn`**：DDPM 的前向假設是 $x_t|x_0\sim N(\sqrt{\bar\alpha}x_0,(1-\bar\alpha)I)$；換成 $\Sigma$ 後 posterior 係數與 score 和 $\hat\varepsilon$ 的關係（會多一個 $\Sigma^{-1}$）都不再成立。資料變換法避開了全部重推。

**只作用於 10 個 rclose 通道**：rhigh/rlow 有符號約束（$r_{high}\ge0\ge r_{low}$），白化會破壞、著色不保證還原。

**實作前的風險驗證**：原本預期「線性組合會因中央極限效應變薄尾」。實測相反——Hill α 由 2.610 變 2.470（更厚）、偏態 −0.446 變 −0.609（更負）。原因是 Cholesky 白化是**逐次剝除已解釋成分**（$z_1=r_1$ 完全不變，$z_k$ 是第 k 檔扣掉前 k−1 檔的迴歸殘差），保留個股特異衝擊，而非取平均。

**無洩漏證據**：$L$ 只由當期訓練窗估計。訓練窗**內**白化後相關 = +0.0000，訓練窗**外** = +0.0305——正因為 $L$ 對後續資料沒有完美白化能力。

### 4.2 第三層：GARCH 遞迴時變波動（波動聚集）

**問題定位**：以 Ljung-Box $Q(20)$ 檢定 $r^2$ 的自相關（對 i.i.d. 有檢定力，而原評估框架的 VC 指標沒有）：

| Series | LB Q(20) | 拒絕「無聚集」 |
|---|---|---|
| REAL | **455** | 10/10 |
| Gaussian null | 39 | 6/10 |
| v-prediction | **43** | 6/10 |
| x_start baseline | **1531** | 10/10 |

v-prediction 的波動聚集**完全消失**（與 i.i.d. 高斯無異）；x_start baseline 反而**過強**（3.4 倍），因為它是條件的平滑確定性函數，繼承了 condition 的波動結構。

**注意**：原框架的 VC 指標給 baseline 0.218、v-pred 0.349——順序與實際相反。該指標計算 $|ACF_{syn}-ACF_{real}|$ 的平均，一個過高、一個為零時誤差絕對值相近，**無法區分「太多」與「沒有」**。

**機制**：$r_t = \sigma_t z_t$，$\sigma_t$ 由 GARCH(1,1) 遞迴。訓練前 $z=r/\sigma(\text{真實過去})$；生成後 $r=\sigma(\text{已生成過去})\cdot z$。

**為什麼用 GARCH 而非 EWMA**：EWMA 的 $\sigma^2$ 是鞅（等價 $\alpha+\beta=1$），生成時遞迴會發散——實測 $\lambda=0.94$ 使 LB Q 衝到 2611。GARCH(1,1) 在 $\alpha+\beta<1$ 時有穩態 $\omega/(1-\alpha-\beta)$，會均值回歸。

**為什麼 pooled 估計**：逐檔 QMLE 的持續性估計雜訊極大（$\alpha+\beta$ 逐檔範圍 0.40–0.997，平均被拉低到 0.85），使 $\sigma_t$ 均值回歸過快、做不出聚集。共用 $\alpha,\beta$（$\omega$ 逐檔）將觀測數由 1,260 提升到 12,600。

| 估計方式 | 遞迴 × 高斯 z 的 LB Q |
|---|---|
| 逐檔 | 94 |
| **pooled** | **496**（真實 455） |

**擬合器驗證**：以模擬的已知 GARCH 做參數回收，$\alpha+\beta$ 誤差僅 0.005–0.013——擬合器本身無誤，先前的低估來自估計雜訊。

**方法論優勢**：生成時的 $\sigma_t$ **完全由模型自己的輸出遞迴產生**，只有 $(\omega,\alpha,\beta)$ 來自訓練窗。生成過程不接觸任何真實資料，因此不存在「看答案」的疑慮（此點強於白化層）。

### 4.3 第四層：Student-t 邊際校準（厚尾）

**先確認波動縮放不足以產生厚尾**：真實報酬經 $\sigma_t$ 標準化後，LB Q 由 455 降到 **5**（時序結構被完整吸收），但超額峰度由 10.70 **升到 14.73**——厚尾主要來自殘差本身，不是波動混合。

由 $\mathrm{df}=6/\text{exKurt}+4$ 反推真實殘差的 Student-t 自由度：$\mathrm{df}\approx 4.4$–4.8（**由資料估出，非調參**）。

**實作方式：保序分位數映射**（rank → uniform → t 分位數），而非修改擴散的噪聲先驗。

**理由**：DPM-Solver 是確定性 ODE，輸出是 $x_T$ 的確定性函數——把 $x_T$ 換成 Student-t **不保證**輸出邊際也是 Student-t。分位數映射是直接且可驗證的，且**單調變換完全保留 copula**（跨資產相依與時序順序都不變）。

實測驗證：套用後平均相關 0.391 → 0.378（幾乎不變），Hill α 3.885 → 3.041、超額峰度 0.75 → 4.15。

### 4.4 跨資產相關的二次修正

四層合併後 Cross-Asset 掉到 0.242。**逐步定位**：

| Model | 還原**前** z 的相關 | 還原**後** r 的相關 |
|---|---|---|
| v + 白化 | 0.408 | 0.408 |
| v + 白化 + 波動縮放 | **0.311** | 0.295 |

相關在**還原前就已經掉了**，還原步驟只再損失 0.016。問題在白化矩陣的估計對象：

| 訓練窗 | corr(r) | corr(z=r/σ) | 落差 |
|---|---|---|---|
| cycle 0 | 0.290 | 0.276 | 0.014 |
| cycle 14（含 COVID） | 0.447 | 0.313 | **0.133** |
| cycle 21 | 0.449 | 0.349 | 0.101 |
| cycle 27 | 0.398 | 0.296 | 0.102 |

**真實報酬的相關有一部分來自共同波動**——市場動盪時所有資產同步大幅波動。逐檔以自身 $\sigma_t$ 標準化會移除此成分，且**波動期落差最大**。生成時著色只還原 $\mathrm{corr}(z)$，而獨立的 $\sigma$ 路徑補不回共同波動的貢獻。

**修正**：白化與著色使用**不同矩陣**——白化用 $\mathrm{chol}(\mathrm{cov}(z))$ 剝乾淨，著色用 $\mathrm{chol}(\mathrm{corr}(r))$ 注入目標相關。兩者刻意不互為反矩陣，因為中間隔著一個會移除共同波動的標準化步驟。

> **被否決的替代方案**：曾嘗試在 GARCH 的 shock 項混入市場共同衝擊。實測該作法將 LB Q 由 416 砍到 161（共同驅動稀釋了個股的波動持續性），相關僅提升 0.010。程式碼保留為 `--vol_common`（預設關閉）供 ablation。

---

## 5. 結果

### 5.1 Overall（primary window 2018-2024，唯一決定排名的期間）

| Method | Stylized Facts | Cross-Asset | **Overall** |
|---|---|---|---|
| **v + 4-Layer (corr-fixed)** | 0.4374 | 0.5145 | **0.4759** |
| v + Cross-Asset Whitening | 0.3137 | **0.6095** | 0.4616 |
| **[NULL] Gaussian(Σ̂)** | 0.3142 | 0.5771 | **0.4457** |
| v + Whiten + VolScale + HeavyTail | **0.4414** | 0.2419 | 0.3417 |
| **[NULL] GARCH+Σ+t** | 0.4238 | 0.3044 | 0.3641 |
| NS-Diff | 0.2802 | 0.3980 | 0.3391 |
| v + Whiten + VolScale | 0.3398 | 0.2712 | 0.3055 |
| Baseline (TimeDiff, x_start) | 0.2383 | 0.3754 | 0.3069 |
| ARMD | 0.2917 | 0.2140 | 0.2529 |
| CN-Diff | 0.1642 | 0.3102 | 0.2372 |
| v + Cross Loss | 0.3235 | 0.0728 | 0.1982 |
| TimeDiff (v-prediction) | 0.3208 | 0.0255 | 0.1732 |
| v + High-Noise Cross Loss | 0.2430 | 0.1003 | 0.1717 |

### 5.2 TABLE SF-1（10 檔平均 normalized score，1 = 與真實一致）

| Method | Heavy Tail | Neg. Skew | Jump | Vol. Cluster | Leverage | **SF Score** |
|---|---|---|---|---|---|---|
| v + Whiten+VolScale+HeavyTail | **0.491** | 0.494 | **0.404** | 0.396 | 0.372 | **0.441** |
| **v + 4-Layer (corr-fixed)** | 0.478 | 0.491 | 0.404 | **0.397** | 0.369 | **0.437** |
| [NULL] GARCH+Σ+t | 0.430 | 0.489 | 0.384 | 0.396 | 0.384 | 0.424 |
| v + Whiten + VolScale | 0.171 | 0.519 | 0.268 | 0.380 | 0.357 | 0.340 |
| v + Cross Loss | 0.094 | 0.504 | 0.198 | **0.419** | **0.428** | 0.324 |
| TimeDiff (v-prediction) | 0.071 | 0.541 | 0.237 | 0.349 | 0.420 | 0.321 |
| [NULL] Gaussian(Σ̂) | 0.068 | 0.530 | 0.233 | 0.345 | 0.411 | 0.314 |
| Baseline (x_start) | 0.133 | **0.594** | 0.006 | 0.218 | 0.115 | 0.238 |

### 5.3 Cross-Asset

| Method | Corr MAE | Corr Frobenius | PCA Dist (TV) | Tail Dep Dist | **CA Score** |
|---|---|---|---|---|---|
| v + Cross-Asset Whitening | **0.0283** | **0.334** | **0.0181** | 0.1097 | **0.6095** |
| **v + 4-Layer (corr-fixed)** | 0.0395 | 0.459 | 0.0344 | 0.1288 | **0.5145** |
| [NULL] Gaussian(Σ̂) | 0.0341 | 0.395 | 0.0266 | **0.1019** | 0.5771 |
| NS-Diff | 0.0674 | 0.775 | 0.0552 | 0.1396 | 0.3980 |
| Baseline (x_start) | 0.0787 | 0.891 | 0.0704 | 0.1441 | 0.3754 |
| [NULL] GARCH+Σ+t | 0.0924 | 0.955 | 0.0827 | 0.1557 | 0.3044 |
| TimeDiff (v-prediction) | 0.3986 | 3.954 | 0.3562 | 0.2602 | 0.0255 |

### 5.4 原始統計量（未經 score 轉換，便於直接判讀）

| Series | Hill α | exKurt | LB Q(20) | 平均相關 | VolRatio |
|---|---|---|---|---|---|
| **REAL** | **2.610** | **10.70** | **455** | **0.396** | **1.000** |
| v + 4-Layer (corr-fixed) | 2.781 | 6.13 | 277 | 0.368 | 0.902 |
| [NULL] GARCH+Σ+t | 2.913 | 7.34 | 349 | 0.304 | 0.938 |
| v + Whiten + VolScale | 3.712 | 1.35 | 519 | 0.295 | 0.949 |
| Baseline (x_start) | 3.862 | 12.75 | 1531 | 0.363 | 0.229 |
| TimeDiff (v-prediction) | 4.318 | 0.19 | 43 | −0.003 | 0.933 |
| [NULL] Gaussian(Σ̂) | 4.380 | 0.17 | 39 | 0.422 | 0.931 |
| NS-Diff | 3.504 | 2.79 | 1144 | 0.349 | 0.387 |
| ARMD | 2.379 | 8.74 | 945 | 0.235 | 0.120 |
| CN-Diff | 3.025 | 6.22 | 12595 | 0.387 | 0.218 |

### 5.5 逐層貢獻

| 層 | 機制 | 主要改善 |
|---|---|---|
| 1. v-parameterization | $x_t$ 恆等路徑內建，係數自帶 t 依賴 | VolRatio 0.229 → 0.933；denoiser 斜率 2.6% → 99.3% |
| 2. Cholesky 白化 + 分離著色 | 線性變換等價於相關噪聲先驗 | Corr MAE 0.399 → 0.040 |
| 3. GARCH 遞迴 $\sigma_t$（pooled） | 時變尺度，由生成路徑自我推進 | LB Q 43 → 519（真實 455） |
| 4. Student-t 邊際校準 | 保序分位數映射，保留 copula | Heavy Tail score 0.068 → 0.478 |

---

## 6. Null Model 與貢獻歸屬（必須誠實呈現的部分）

為避免高估模型貢獻，建立了兩個**不含任何生成模型**的對照，走完全相同的 28 期滾動協定、所有參數只用訓練窗估計：

| Null Model | 組成 |
|---|---|
| Gaussian(Σ̂) | 訓練窗的 mean/cov + 多元高斯 |
| GARCH+Σ+t | 訓練窗的 GARCH(1,1) + Cholesky 相關 + Student-t 殘差（與四層結構完全對應） |

**結果**：

| | SF Score | CA Score | Overall |
|---|---|---|---|
| v + 4-Layer (corr-fixed) | 0.437 | 0.515 | **0.476** |
| [NULL] Gaussian(Σ̂) | 0.314 | 0.577 | 0.446 |
| [NULL] GARCH+Σ+t | 0.424 | 0.304 | 0.364 |

**應誠實陳述的三點**：

1. **最佳模型僅險勝 Gaussian null 0.030**。若未納入 null model 對照，容易得出過度樂觀的結論。
2. **拆解貢獻來源**：
   - Stylized Facts 的改善（0.314 → 0.437）**主要來自結構層**——GARCH-null 單靠結構就達 0.424。
   - Cross-Asset 上，四層版 0.515 顯著高於 GARCH-null 的 0.304——**這部分是 diffusion 模型 + 白化的貢獻**。
3. **Gaussian null 在 Cross-Asset 反而最高（0.577）**。因為它直接使用訓練窗的 $\Sigma$，而相關矩陣在時間上高度持續。此指標主要測量的是**相關矩陣的持續性**，而非模型能力。

**綜合結論**：四層結構貢獻了 stylized facts，diffusion + 白化貢獻了跨資產結構，兩者疊加才首次全面超越平凡基準。

---

## 7. 限制

1. **結構性先驗而非學習成果**。模型負責的只剩「給定 condition 生成標準化殘差」。應定位為**混合式生成器**（結構化市場模型 + diffusion 負責條件動態），而非「diffusion 學會了市場特性」。

2. **相關結構被凍結**。$\Sigma$ 每 63 個交易日更新一次，無法重現危機期間的相關飆升。這也是 Tail Dependence 改善有限（0.260 → 0.129，仍差於 Gaussian null 的 0.102）的原因——尾部相依需要**時變**的相關結構。

3. **根本原因未解**。denoiser 仍不會把獨立噪聲映射成相關輸出，相關性外包給固定線性映射。

4. **評估框架的鑑別力不足**。i.i.d. 高斯在 Volatility Clustering 得 0.345、Leverage 得 0.411（訊噪比分別為 1.87 與 0.89，框架已標記為 low statistical power）。建議改報**相對 null model 的 skill score**：$\text{Skill}=(D_{null}-D_{model})/D_{null}$，僅達高斯水準時 skill = 0。

5. **未涵蓋下游任務**。若目標為投資組合配置或 ETF 輪動 RL，最具說服力的評估是：以合成資料建構組合，於真實資料回測，檢驗 VaR/CVaR 的實際突破率。高斯必然低估尾部風險，此處才是生成模型不可被取代之處。

6. **統計功效**：Negative Skewness 與 Leverage Effect 在 primary window 的訊噪比分別為 1.00 與 0.89，接近或低於 1，評估表已標記。

7. **單一 seed**：所有完整訓練均為 `gen_seed=1` 的單次執行，未做多 seed 重複。

---

## 8. 程式碼變更

**擴散核心**（`diffusion_worker.py` / `DDPM.py` / `dpm_sampler.py`）僅為第一層 v-parameterization 而修改；第二至四層**完全在 `walk_forward_dow10.py` 內以資料變換實作**，未觸及擴散數學。

| 檔案 | 變更 |
|---|---|
| `models_diffusion/diffusion_worker.py` | v-parameterization（`get_v`、`predict_start_from_v`、target/assert/lvlb 分支）；cross-asset loss 掛載點；energy score；`_aux_x0_hat` |
| `models_diffusion/DDPM.py` | 放寬 sampler 的 parameterization assert（1 行） |
| `models_diffusion/samplers/dpm_sampler.py` | 傳遞 `model_type` 與 `predict_x0`（3 行） |
| `models_diffusion/stylized_loss.py` | `cross_asset_corr_loss`、`energy_score`（附加於檔尾，未動既有函式） |
| `walk_forward_dow10.py` | §2.5 白化、§2.6 GARCH 波動縮放與邊際校準、`--force_step`、CLI 參數 |
| `evaluation/` | `Corr Frobenius` 指標、圖檔改用 tag 命名、null model 納入 |

**新增 CLI**（全部預設關閉，不影響原流程）：

```
--parameterization v          第一層
--whiten_cross_asset          第二層
--vol_scaling                 第三層
--heavy_tail_residual         第四層
--vol_common                  （ablation，實測有害）
--lambda_cross_max            （cross-asset loss，實測效率低）
--aux_no_teacher_forcing      （繞過 future mixup）
--cross_gen_aligned           （繞過 x_t 的 x0 殘留）
--force_step                  （控制實驗用，跳過選模階段）
```

**重現指令**：

```bash
python walk_forward_dow10.py --tag v_4layer_fix \
  --max_steps 600 --eval_every 50 --val_batches 4 \
  --n_paths 50 --gen_seed 1 --save_ckpt --long_format \
  --parameterization v --whiten_cross_asset --vol_scaling --heavy_tail_residual

python -m evaluation.run_all
```

---

## 9. 建議的後續方向

1. **提升評估鑑別力**（優先）：改報相對 null model 的 skill score；VC/Leverage 改用對 i.i.d. 有檢定力的統計量（如 Ljung-Box Q）。在評估工具能區分「有無該性質」之前，任何模型改動都無法被驗證。

2. **時變相關結構**：現行 $\Sigma$ 逐期固定。若能讓相關隨波動狀態變化（如 DCC-GARCH 的結構），可望改善 Tail Dependence——那是目前唯一仍輸給 Gaussian null 的 cross-asset 指標。

3. **下游任務導向評估**：以投資組合回測（VaR/CVaR coverage、Sharpe、max drawdown）取代純分布指標。此評估難以被平凡基準取代，且直接對應實際用途。

4. **條件生成的價值驗證**：四層結構把統計性質固定後，diffusion 唯一無法被 GARCH/高斯取代的能力是**條件生成**——給定不同市場情境的 condition，產生對應的市場路徑。這正是 ETF 輪動 RL 所需（agent 需在可指定、多樣化的市場情境上訓練），也是最能凸顯 conditional diffusion 價值的定位。

---

## 附錄：評估框架

- **Primary window** 2018-01-02 ~ 2024-12-31（T=1,761）於任何模型分數計算前即已固定，選擇依據為真實市場資料的 regime 覆蓋與訊噪比，非模型表現。
- **Score 轉換**：$\text{Score}=2^{-D/\tau}$，$\tau$ 為真實資料 stationary block bootstrap 的重抽噪音尺度（$D=\tau \Rightarrow$ score 0.5）。$\tau$ 僅由真實資料計算，不接觸任何 synthetic 輸出。
- **Heavy Tail 的 $\tau$** 改用不重疊的 m-out-of-n subsampling（一般 bootstrap 對極值順序統計量不一致，Athreya 1987）。
- 詳細方法論見 `results/evaluation_methodology_report.txt`、`results/TABLE_SF1_methodology.md`、`results/TABLE_CrossAsset_methodology.md`。
