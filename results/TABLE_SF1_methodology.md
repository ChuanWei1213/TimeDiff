# TABLE SF-1 — Distance 與 Normalization 的設計說明

## 0. 核心問題：為什麼不能五個指標都用 $|{\rm syn}-{\rm real}|$

五個 stylized facts 雖然都是「合成資料與真實資料的差距」，但它們**活在不同的數學空間**：

| Fact | 這個量是什麼 | 值域 | 可以跨零嗎 |
|---|---|---|---|
| Heavy Tail | 正的**尺度參數** $\alpha$ | $(0,\infty)$ | 否 |
| Negative Skewness | **無量綱**的標準化三階動差 | $(-\infty,\infty)$ | 是 |
| Jump | 一個**比例** + 一個**幅度** | $[0,1]$ 與 $(0,\infty)$ | 否 |
| Volatility Clustering | 一條**曲線** $\rho(k),\ k=1..K$ | 每點 $[-1,1]$ | 是 |
| Leverage Effect | 一條**曲線** $L(k),\ k=1..K$ | 每點 $[-1,1]$ | 是 |

對它們套用同一個 $|{\rm syn}-{\rm real}|$ 會產生三種具體錯誤：

1. **尺度參數用絕對差會失真**。$\alpha$ 從 2 變 3，意味著尾部從「四階動差不存在」變成「存在」——
   在風險上是質變；$\alpha$ 從 6 變 7，兩者都已是輕尾，實務上沒有差別。
   兩者的絕對差都是 1，但金融意義天差地遠。
2. **曲線壓成單一數字會漏掉形狀**。兩個模型可以有相同的 $\rho(1)$，但一個緩慢衰減（像真實市場的長記憶）、
   另一個在 $k=3$ 就掉到 0。只比 lag 1 會給後者滿分。
3. **不同單位的量不能相加**。Jump frequency 的單位是機率、magnitude 是無量綱的對數比，
   直接加權平均等於把兩個不同單位的東西相加。

因此按數學性質分成三類，各用適當的 distance：

```
distribution-level : Heavy Tail, Negative Skewness      → 分布參數的差距
event-level        : Jump Behaviour                     → 事件的頻率與幅度
temporal-dynamics  : Volatility Clustering, Leverage    → 整條自相關曲線的差距
```

---

## 1. 每個 Metric 用自己的 Distance

### 1.1 Heavy Tail — log-ratio distance

**Raw metric**：左尾 Hill 尾部指數。設 $X$ 為由大到小排序的 $\{-r:\ r<0\}$，
$k=\lceil n_{\rm neg}^{2/3}\rceil$，門檻 $u=X_{(k)}$：

$$P(X>x)\sim x^{-\alpha},\qquad
\hat\alpha=\Big[\tfrac1k\sum_{j=1}^{k}\log\tfrac{X_{(j)}}{u}\Big]^{-1}$$

$\alpha$ 越小 ⇒ 尾巴越厚。只用負報酬，因為下跌尾部才是金融風險關心的對象。

**Distance**

$$D_{\rm tail}=\big|\log(\alpha_{\rm syn}/\alpha_{\rm real})\big|$$

**為什麼是 log-ratio**

- $\alpha$ 是**正的尺度參數**，其估計量近似對數常態。對尺度參數而言，「差幾倍」才是有意義的比較單位，「差多少」不是。
- **對稱性**：高估兩倍（$\alpha_{\rm syn}=2\alpha_{\rm real}$）與低估兩倍（$\alpha_{\rm syn}=\alpha_{\rm real}/2$）
  應該被視為同樣嚴重。log-ratio 兩者都給 $\log 2=0.693$。
  若改用相對誤差 $|\Delta\alpha|/\alpha_{\rm real}$，前者得 1.0、後者得 0.5 ——
  **系統性偏袒低估**，也就是偏袒「尾巴做得太厚」的模型。這是不可接受的偏誤。
- $\alpha>0$ 有下界且遠離 0，取 log 不會遇到除以零的問題。

### 1.2 Negative Skewness — 絕對距離 + 獨立的符號診斷

**Raw metric**：$g_1=m_3/m_2^{3/2}$（小樣本偏誤修正）。

**Distance**

$$D_{\rm skew}=\big|{\rm skew}_{\rm syn}-{\rm skew}_{\rm real}\big|$$

**為什麼這裡反而是絕對距離**

- Skewness 本身**已經是標準化過的量**（除以 $\sigma^3$），無量綱，不同股票之間本來就可比，
  不需要再做尺度轉換。
- 它**可正可負、可以跨零**，所以 log-ratio 無定義、相對誤差在真實值接近 0 時會爆炸。
  這正是 1.1 的作法在這裡不能用的原因 —— 兩者的差異來自量本身的性質，不是任意選擇。

**符號一致性單獨呈現（不併入分數）**

$$A_{\rm skew}=\frac1{10}\sum_i \mathbb{1}\big[{\rm sign}({\rm skew}_{{\rm syn},i})={\rm sign}({\rm skew}_{{\rm real},i})\big]$$

理由：真實 $=-0.95$、合成 $=+0.10$ 不只是「數值差 1.05」，而是**方向完全相反** ——
模型把「大跌比大漲更常見」這個事實反過來學了。
但若要把符號錯誤併進分數，就必須引入一個懲罰係數 $\gamma$，而 $\gamma$ 沒有任何理論依據，
屬於任意設定。因此改為**獨立欄位呈現**。
實際上絕對距離已經隱含懲罰了符號錯誤（$1.05$ 相對於 $\tau_{\rm skew}$ 是很大的距離），
不需要額外機制。

### 1.3 Jump Behaviour — 先定義事件，再比頻率與幅度

Jump 不是一個可以直接相減的統計量，必須**先定義什麼叫一次 jump**。

**門檻由真實資料定義**

$$u_i=q_{0.95}\big(|r^{\rm real}_i|\big)\quad\text{（逐檔）},\qquad
\text{synthetic 一律套用同一個 } u_i$$

**這是整個設計中最關鍵的一點。** 若讓 synthetic 用自己的 $q_{0.95}$，
則一條波動只有真實 1/5 的序列也會「剛好」有 5% 的觀測超過它自己的 $q_{0.95}$ ——
**任何模型都會自動得到正確的 jump frequency**，指標完全失效。
這叫 circular evaluation：用被評估對象自己的分布去定義評估標準。

**兩個子指標**

$$f_i=\frac{\#\{t:\ |r_{t,i}|>u_i\}}{T},\qquad
m_i=\mathbb E\big[\,|r_{t,i}|\ \big|\ |r_{t,i}|>u_i\,\big]$$

$$D_{\rm freq}=\big|f_{\rm syn}-f_{\rm real}\big|,\qquad
D_{\rm mag}=\big|\log(m_{\rm syn}/m_{\rm real})\big|$$

- **Frequency 用差**：$f$ 是比例，兩個比例的差本身就是有意義的量（差 1 個百分點就是差 1 個百分點）。
- **Magnitude 用 log-ratio**：$m$ 是正的尺度參數，理由同 1.1。

**先各自轉成分數，再等權平均**

$$S_{\rm jump}=\tfrac12\,2^{-D_{\rm freq}/\tau_{\rm freq}}+\tfrac12\,2^{-D_{\rm mag}/\tau_{\rm mag}}$$

**不在 distance 層合併**：$D_{\rm freq}$ 的單位是機率、$D_{\rm mag}$ 是無量綱對數比，
直接加權平均會把兩個不同單位的量相加。必須先各自無量綱化成分數，才能合併。

**等權的理由**：複合 Poisson 跳躍模型 $\mathrm dr_t=\sigma\,\mathrm dW_t+J_t\,\mathrm dN_t$
的跳躍成分由兩個參數完全刻畫 —— 強度 $\lambda$（對應 frequency）與跳躍幅度分布（對應 magnitude）。
沒有理論說哪一個更重要，故等權。

**無超越事件的處理**：若合成序列在真實門檻下一次都沒超過，magnitude 未定義。
此時 magnitude 子分數記為 0，並在表格標記 `no exceedance`，
**不可當成缺失值丟棄** —— 那會讓表現最差的模型從平均中消失。

### 1.4 Volatility Clustering — L1 曲線距離

**Raw metric**：以 $r^2$ 為波動代理（對應 realized variance，與 GARCH／BNS 文獻一致），

$$\rho(k)=\mathrm{Corr}\big(r_t^2,\ r_{t+k}^2\big),\qquad k=1,\dots,K,\quad K=20$$

**Distance**

$$D_{\rm VC}=\frac1K\sum_{k=1}^{K}\big|\rho_{\rm syn}(k)-\rho_{\rm real}(k)\big|$$

**為什麼比整條曲線而不是只比 lag 1**

波動聚集的定義是「波動的自相關為正**且緩慢衰減**」。單一 lag 只檢驗了前半句。
兩個模型可以有完全相同的 $\rho(1)$，但一個像真實市場那樣緩慢衰減（長記憶）、
另一個在 $k=3$ 就掉到 0。只比 lag 1 會給後者滿分。
比較整條曲線同時檢驗了**水平**（clustering 有多強）與**衰減形狀**（記憶有多長）。

**為什麼用 L1 而不是 L2**

L2 會讓單一 lag 的大偏差平方後主導整條曲線的距離；L1 對個別 lag 的離群值穩健，
反映的是「整條曲線平均偏離多少」，與我們想問的問題一致。

**為什麼 $K=20$**（三個獨立理由）

1. **經濟意義**：約一個交易月，是波動聚集在風險模型與選擇權定價上最相關的水平。
2. **統計意義**：Bartlett 標準誤約 $1/\sqrt T$；更長的 lag 上真實 ACF 已落在噪音帶內，
   再比下去只是在比較噪音。
3. **生成協定的硬限制**：walk-forward 每 21 個交易日重新錨定到一段真實 condition。
   因此 $k\ge 21$ 的相依性**有一部分來自真實 condition 而非模型本身**，
   納入評估會系統性高估模型。故必須 $K<21$。

### 1.5 Leverage Effect — L1 曲線距離 + 獨立的方向診斷

**Raw metric**

$$L(k)=\mathrm{Corr}\big(r_t,\ r_{t+k}^2\big),\qquad k=1,\dots,K,\quad K=20$$

真實市場通常為**負**：今日下跌 → 未來波動上升。

**Distance**

$$D_{\rm LE}=\frac1K\sum_{k=1}^{K}\big|L_{\rm syn}(k)-L_{\rm real}(k)\big|$$

（L1 曲線距離與 $K=20$ 的理由同 1.4。）

**方向單獨診斷**：報告 (a) $\mathrm{sign}\big(\overline{L_{\rm syn}(k)}\big)$ 是否為負；
(b) 逐 lag 的符號一致率。
理由同 1.2 —— 一個模型可能數值距離不大但符號相反，那代表它學到的是
「下跌後波動下降」，與真實市場的機制相反。方向與數值距離分開報告，
避免引入沒有依據的符號懲罰係數。

### 1.6 五個 Distance 對照總表

| Fact | 類型 | Raw metric | Distance $D$ | 選擇這個 $D$ 的關鍵理由 |
|---|---|---|---|---|
| Heavy Tail | distribution | Hill $\hat\alpha$ | $\lvert\log(\alpha_{\rm syn}/\alpha_{\rm real})\rvert$ | 正尺度參數；對高估／低估對稱 |
| Negative Skewness | distribution | $g_1$ | $\lvert{\rm skew}_{\rm syn}-{\rm skew}_{\rm real}\rvert$ | 已無量綱且可跨零，比值無定義 |
| Jump Behaviour | event | $(f,m)$ @ 真實門檻 | $\lvert\Delta f\rvert$ 與 $\lvert\log(m_{\rm syn}/m_{\rm real})\rvert$ | 比例用差、幅度用比；門檻必須來自真實資料 |
| Volatility Clustering | temporal | $\rho(k),k\le20$ | $\frac1K\sum_k\lvert\Delta\rho(k)\rvert$ | 須同時檢驗水平與衰減形狀 |
| Leverage Effect | temporal | $L(k),k\le20$ | $\frac1K\sum_k\lvert\Delta L(k)\rvert$ | 同上，另加方向診斷 |

---

## 2. Normalization：以真實資料的重抽噪音為尺度

### 2.1 為什麼一定要做這一步

上一節得到五個 $D$，但它們的**單位完全不同**：

$$D_{\rm tail}\ \text{是對數比}\quad
D_{\rm skew}\ \text{是無量綱}\quad
D_{\rm freq}\ \text{是機率}\quad
D_{\rm VC},D_{\rm LE}\ \text{是相關係數的平均絕對差}$$

$D=0.1$ 對其中一個可能是嚴重偏離，對另一個可能微不足道。
**這些數字不能互相比較，更不能直接平均成一個 SF Score。**
必須先把每個 $D$ 換算成一個共同的、無量綱的尺度。

### 2.2 τ 的定義

用**真實資料自身的估計噪音**當作那把尺：

$$\tau_m=\mathbb E_b\Big[D_m\big(\hat\theta(X^{(b_1)}_{\rm real}),\ \hat\theta(X^{(b_2)}_{\rm real})\big)\Big]$$

**做法**：把真實序列用 stationary block bootstrap 重抽兩次，得到兩組獨立的重抽樣本，
各自計算指標值，算兩者的距離；重複 $B=500$ 次取平均。

**直覺**：即使有一個完美的模型，它產生的資料與真實資料之間也不會距離為零 ——
因為真實資料本身只是隨機過程的一次實現，重抽一次就會得到不同的估計值。
$\tau$ 量化的就是**這個無法消除的隨機性下限**。
如果合成資料的誤差小於 $\tau$，代表它已經做得跟「真實股市自己跟自己比」一樣好。

**為什麼用 stationary block bootstrap 而不是逐點重抽**：
逐點 i.i.d. 重抽會把時序結構完全打散，波動聚集與 leverage effect 會消失，
算出來的 $\tau$ 會嚴重低估。block bootstrap 以整段連續區塊重抽
（區塊長度服從幾何分布、平均 $T^{1/3}$），保留時序相依。

**關鍵性質：$\tau$ 只使用真實資料**，計算過程從未接觸任何模型的輸出。
因此不存在「用被評估對象決定評估標準」的循環問題，
也不可能為了讓某個模型好看而調整。

**一個例外，以及它的代價**：Heavy Tail 的 $\tau$ 改用不重疊的 m-out-of-n subsampling。
因為 Hill 是**極值順序統計量**，一般 bootstrap 對它不一致（Athreya 1987）——
重抽會重複抽到同一批極端觀測，低估尾部的真實變異。

作法是在整段期間裡切出兩段**互不重疊**、長度皆為 $m$ 的子區間，
在多個 $m$ 上計算 $\tau(m)$，擬合 $\log\tau=a+b\log m$ 後外推到 $n$。
由 $k=\lceil n^{2/3}\rceil$ 可推得 $b=-1/3$，實證擬合亦得到相符的值。

必須誠實揭露的代價：**兩段不重疊的子區間必然來自不同的時間段**
（例如拿 2020 COVID 那段與 2023 平靜期相比），
因此 $\tau_{\rm heavy\ tail}$ 同時包含了抽樣噪音**與 regime 之間的真實差異**，
會比純粹的抽樣噪音大，使 Heavy Tail 的評分偏寬鬆。
這是兩害相權的結果 —— 另一個選項（一般 bootstrap）在理論上就是錯的。
由於 $\tau$ 與模型無關，**Heavy Tail 一欄的模型排名不受影響**，只有絕對水準偏高。

### 2.3 Score 轉換

$$\boxed{\ \text{Score}_m=2^{-D_m/\tau_m}\ \in(0,1]}$$

| $D/\tau$ | Score | 解讀 |
|---|---|---|
| 0 | 1.000 | 與真實完全一致 |
| 0.5 | 0.707 | 誤差只有自然噪音的一半 |
| **1** | **0.500** | **誤差恰等於真實資料自身的重抽誤差** |
| 2 | 0.250 | 誤差是自然噪音的 2 倍 |
| 3 | 0.125 | 3 倍 |
| 4 | 0.063 | 4 倍 |

分數的意義因此非常明確：**「這個模型的誤差，是真實資料自身隨機性的幾倍」**，
每多一倍，分數減半。

### 2.4 為什麼不用其他常見的轉換

| 候選 | 問題 |
|---|---|
| $1-D$ | 需要 $D\in[0,1]$。五個 $D$ 中只有 PCA 的 TV distance 天然有界，其餘無上界，會出現負分 |
| $e^{-D}$ | 形式上可行，但等於把尺度隱含設為 $\tau=1/\ln 2$ —— 是一個**任意**的常數，沒有任何依據 |
| $D/\lvert\theta_{\rm real}\rvert$（相對誤差） | **近零分母問題**：leverage effect 的真實值約 $-0.03$、jump frequency 為 $0.05$，除以一個接近 0 的數會產生假的巨大相對誤差。本框架除以**標準誤**而非 metric 值，因此近零不構成問題 |

### 2.5 τ 的計算粒度

$\tau$ 不是一個全域常數，而是分層計算：

- **逐 window**：primary（2018–2024）與各 robustness window 各自用**該期間**的真實資料計算。
  樣本從 1,761 天降到約 252 天時 $\tau$ 幾乎翻倍（噪音 $\propto 1/\sqrt T$），
  短窗的容忍度因此自動放寬 —— 否則短窗會被不公平地打低分。
- **逐 ticker（univariate 五個指標）**：每一檔股票的每一個指標都有自己的 $\tau$，
  共 $10\times5=50$ 個值。波動大的股票估計本來就比較不穩，
  用共同的 $\tau$ 會系統性偏袒某些股票。
- **Cross-asset 三個指標**：本質上是 $10\times10$ 矩陣層級的量，沒有「逐檔」的概念，
  故每個 window 一個 $\tau$；重抽時**整列一起抽**以保留橫斷面相依。

---

## 3. 聚合方式

$$\text{for each ticker } i:\quad
\theta_{{\rm real},i},\ \theta_{{\rm syn},i}\ \to\ D_{m,i}\ \to\ S_{m,i}=2^{-D_{m,i}/\tau_{m,i}}$$

$$S_m=\frac1{10}\sum_{i=1}^{10}S_{m,i}$$

**必須逐檔算完再平均，不可先把 10 檔資料串接起來算。** 串接會製造 9 個假的
「跨股票斷點」污染所有時序指標（VC、LE），並把不同尺度的資產混成一個混合分布，
使 Hill 與 skewness 變成混合分布的統計量而非單一資產的性質。

每一檔的 $\theta_{\rm real}$、$\theta_{\rm syn}$、$D$、$\tau$、$S$ 全部保留，
供 TABLE SF-3（asset-level dispersion）與附錄使用。

---

## 4. 三張表的分工

| 表 | 內容 | 方向 | 目的 |
|---|---|---|---|
| **SF-1** | 五個 fact 的 fidelity score + SF Score | 越高越好 | 主表，可跨指標比較 |
| **SF-2** | 五個 fact 的 raw distance $D$ | 越低越好 | 讓 reviewer 看到 score 背後的實際誤差，可檢核 |
| **SF-3** | 逐檔 score 的 Mean / Std / Min / Max | — | 回答「平均分數是否只被少數幾支股票拉高」 |

**raw distance 與 score 刻意分成兩張表**，不混在一起 ——
score 經過了 $\tau$ 的無量綱化，是可比的；raw distance 保留原始單位，是可檢核的。
兩者放同一張表會讓讀者誤以為它們可以互相對照。
