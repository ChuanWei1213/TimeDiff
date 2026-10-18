# Research Experiment Report — Financial Time Series Diffusion (DOW10)

> Audit date: 2026-08-09. Scope: DOW10 experiments only. Method: evidence-based
> inventory (checkpoints, outputs, logs, code). Nothing below is assumed; items
> without evidence are marked `NOT VERIFIED` / `✗ MISSING` / `? UNCERTAIN`.
> Results from a different dataset / protocol are marked `NOT DIRECTLY COMPARABLE`.

---

## 0. Scope & Comparability Guard

**Two distinct datasets exist in this project. Do NOT mix them.**

| Dataset | Tickers | Files | Used by |
|:---|:---|:---|:---|
| **DOW10** (this report) | AAPL, AMGN, CRM, CSCO, IBM, INTC, MSFT, NKE, VZ, WMT | `dow10_returns.csv` | all `dow10_*` checkpoints, `diagnostic_dow10_*` |
| Old 5/large-ticker set | NVDA, AAPL, MSFT, GOOGL, AMZN … | `financial_returns.csv` | `diagnostic_baseline`, `diagnostic_vol_lev`, `*0729`, `diagnostic_ablation_pred63` |

> ⚠️ `outputs/diagnostic_baseline`, `diagnostic_vol_lev`, `diagnostic_baseline_VOL2/VOL4/nonorm`,
> `diagnostic_*0729`, `diagnostic_ablation_pred63` are **NVDA-based, NOT DOW10** →
> `NOT DIRECTLY COMPARABLE`. Evidence: `basic_statistics.csv` first ticker = NVDA.

**Canonical DOW10 protocol** (verified from `generate_dow10.py`):
- Train 2010-01-01…2023-12-31; generate full-year 2024 (252 trading days).
- `seq_len=63`, `pred_len=21`, `label_len=21`, `num_vars=30` (rclose/rhigh/rlow × 10).
- **Stylized facts computed on rclose channels ONLY** (ch 0,3,…,27); rhigh/rlow excluded. ✓ verified in both `generate_dow10.py:compute_stylized_facts` and `diagnostic_analysis.py:load_synthetic` (selects `*_rclose`).
- Rolling = **Pure Real Condition**, stride=21, 12 rolls (condition always real history).
- Price reconstruction = **chained generated returns** (`reconstruct_prices`, uses generated previous day, NOT reset per window). ✓ verified `generate_dow10.py:259-267`.
- Parameterization `x_start` (except `dow10_eps`), diffusion steps 100, cosine β schedule.

**Metric protocol used for the reliable table below** (this-session verification, read-only):
Real 2024 DOW10 rclose vs synthetic rclose, aligned to min length; the 5 stylized facts
are **scale-invariant** (Hill, skew, jump-RJ, vol-ACF, leverage), so `vol_scale` does NOT
affect them → all generation configs generated at `vol_scale=1.0` are directly comparable.

**REAL 2024 target values** (the numbers every synthetic run is compared against):

| Hill α | Excess Kurt | Skew | Jump RJ | Vol-ACF(1) | Leverage |
|---:|---:|---:|---:|---:|---:|
| **2.79** | **14.96** | **-0.954** | **0.249** | **0.030** | **-0.029** |

---

## 1. Model Architecture

- **TimeDiff** DDPM, `x_start` parameterization (predict clean x₀), MSE denoising loss.
- Condition encoder: DLinear branch + per-asset linear AR projection.
- Denoiser: `CNN_DiffusionUnet` (`models_diffusion/DDPM_modules/DDPM_CNNNet.py`) —
  x_t → `input_projection` → concat(diffusion-timestep-embed) → `enc_conv`; condition →
  `cond_projections` (AR) + DLinear output; fused by `combine_conv` → predicted x₀.
  **No x_t→output identity skip** (relevant to failure modes, §11).
- Sampler for generation: ancestral (`type_sampler none`), 100 reverse steps, Gaussian noise.

---

## 2. Experiment Registry (DOW10 only)

Fixed for all rows unless noted: Dataset=DOW10, seq=63, pred=21, label=21, rolling=PureReal
stride21 ×12, sampler=ancestral, parameterization=x_start.

| ID | Config (losses / intervention) | λ_vol | λ_lev | λ_tail/other | Epochs* | Checkpoint | Output | Eval | Status |
|:---|:---|:--:|:--:|:--|:--:|:--:|:--:|:--:|:--:|
| `dow10` | plain — **claimed** Diffusion+MSE, no losses | ? | ? | ? | 16 | ✓ | ✗ none | ✗ none | ? UNCERTAIN |
| `dow10_sty1` | VOL+LEV (pre-"close-only"), all-channel | 0.1? | 0.1? | — | 31 | ✓ | `dow10_sty1_flat` | `diagnostic_dow10_sty1_flat` | ⚠ PARTIAL |
| `dow10_sty0.3` | VOL+LEV λ≈0.3 (older) | 0.3? | 0.3? | — | ? | ✓ | ✗ | ✗ | ? UNCERTAIN |
| **`dow10_sty1_c`** | **VOL+LEV, close-only** (de-facto working model) | 0.1 | 0.1 | — | 25 | ✓ | `dow10_sty1_c` (+vs2.0) | `diagnostic_dow10_sty1_c` (full Q1–Q9) | ✓ COMPLETED |
| `dow10_sty1_c_tail` | + Pinball tail | 0.1 | 0.1 | tail=0.3, best-val | ~? | ✓ | ✓ | per-ticker txt | ⚠ PARTIAL (best-val trap) |
| `dow10_up3_q90` | + Extreme-window oversampling 3×, q90 | 0.1 | 0.1 | ratio3/thr0.90 | ? | ✓ | ✓ | re-eval this session | ✓ COMPLETED |
| `dow10_up3_q95` | + oversampling 3×, q95 | 0.1 | 0.1 | ratio3/thr0.95 | ? | ✓ | ✓ | re-eval this session | ✓ COMPLETED |
| `dow10_up5_q95` | + oversampling 5×, q95 | 0.1 | 0.1 | ratio5/thr0.95 | ? | ✓ | ✓ | re-eval this session | ✓ COMPLETED |
| `dow10_sty1_c_res` | + x_t residual skip (early-stop@9) | 0.1 | 0.1 | residual | 19(best9) | ✓ | (probed) | sensitivity+collapse | ✓ COMPLETED |
| `dow10_sty1_c_res_full` | + x_t residual skip (full, save_last) | 0.1 | 0.1 | residual | 100 | ✓ | ✓ | 5-metric | ✓ COMPLETED |
| `dow10_eps` | ε-parameterization (noise) | 0.1? | 0.1? | param=noise | 100 | ✓ | ✓(+clip3/5) | 5-metric | ✓ COMPLETED (failed) |
| `dow10_tail_l03` | + Pinball λ_tail=0.3 (save_last, full) | 0.1 | 0.1 | tail=0.3 | 100 | ✓ | ✓ | 5-metric | ✓ COMPLETED |
| `dow10_tail_l10` | + Pinball λ_tail=1.0 | 0.1 | 0.1 | tail=1.0 | 100 | ✓ | ✓ | 5-metric | ✓ COMPLETED |
| `dow10_tail_l30` | + Pinball λ_tail=3.0 | 0.1 | 0.1 | tail=3.0 | 100 | ✓ | ✓ | 5-metric | ✓ COMPLETED |
| `dow10_gpd_l05` | + GPD/Hill ξ-match λ=0.5 (per-ticker) | 0.1 | 0.1 | gpd=0.5 | 100 | ✓ | ✓ | 5-metric | ✓ COMPLETED |
| `dow10_gpd_l20` | + GPD/Hill ξ-match λ=2.0 | 0.1 | 0.1 | gpd=2.0 | 100 | ✓ | ✓ | 5-metric | ✓ COMPLETED |
| `dow10_studt_df3` | + Student-t noise prior df=3 | 0.1 | 0.1 | noise=t(3) | 100 | ✓ | ✓ | 5-metric | ✓ COMPLETED |
| `dow10_studt_df5` | + Student-t noise prior df=5 | 0.1 | 0.1 | noise=t(5) | 100 | ✓ | ✓ | 5-metric | ✓ COMPLETED |

*Epochs from `losses.pkl` length (val-loss checkpoints) or known save_last runs. `?` = not recoverable (no saved args/logs; git history does not cover dow10 runs).

**Config caveat:** no training `args`/config were saved with checkpoints, and no training
logs persist. For `dow10`, `dow10_sty1`, `dow10_sty0.3` the loss configuration is inferred
from tag-naming convention only → `? UNCERTAIN`. All `_res/_eps/_tail/_gpd/_studt` rows were
run this session with known commands (in `run_*_sweep.sh`).

---

## 3. Baseline Results

**Pure Diffusion+MSE baseline (NO vol/lev/tail/oversampling): `✗ MISSING` as a verified, evaluated experiment.**

Evidence:
- A checkpoint tagged `dow10` exists (16 epochs) but **has no saved config** and **no
  generation/stylized-facts output** (`outputs/dow10/` has no `*_stylized_facts_2024.txt`).
- Its loss configuration cannot be confirmed (no args, no logs) → `? UNCERTAIN`.
- **Every quantitative "baseline" used in this session's comparisons is actually
  `dow10_sty1_c` = Baseline + VOL + LEV**, NOT pure MSE. This must be corrected in the paper.

→ **A clean, evaluated pure-MSE baseline is required and currently does not exist.** (P0, §15.)

---

## 4. Stylized Facts Results (reliable, this-session verification, vol_scale=1.0)

All on DOW10 rclose, 2024, same protocol → directly comparable. `dow10_sty1_c` = Base+VOL+LEV.

| Config | Hill α | Kurt | Skew | Jump RJ | Vol-ACF | Leverage |
|:---|---:|---:|---:|---:|---:|---:|
| **REAL (target)** | **2.79** | **14.96** | **-0.954** | **0.249** | **0.030** | **-0.029** |
| dow10_sty1_c (Base+VOL+LEV) | 5.44 | 0.4 | -0.078 | 0.012 | 0.080 | -0.023 |
| + oversampling up3_q90 | 6.87 | 0.2 | -0.158 | 0.002 | 0.105 | +0.093 |
| + oversampling up3_q95 | 5.85 | 0.2 | -0.013 | 0.000 | 0.141 | -0.005 |
| + oversampling up5_q95 | 7.93 | 0.1 | -0.064 | 0.004 | 0.199 | +0.147 |
| + Pinball λ_tail=0.3 | 6.35 | — | -0.111 | 0.007 | 0.072 | +0.063 |
| + Pinball λ_tail=1.0 | 6.31 | — | -0.105 | 0.012 | 0.036 | +0.019 |
| + Pinball λ_tail=3.0 | 5.94 | — | -0.124 | 0.002 | 0.025 | -0.002 |
| + GPD ξ-match λ=0.5 | 6.48 | — | -0.117 | 0.007 | 0.083 | +0.059 |
| + GPD ξ-match λ=2.0 | 5.39 | 0.2 | +0.018 | 0.017 | 0.044 | -0.012 |
| + Student-t noise df=5 | 6.12 | 0.1 | -0.005 | 0.006 | 0.028 | +0.017 |
| + Student-t noise df=3 | 5.40 | 0.5 | -0.055 | 0.029 | -0.007 | +0.021 |

**Per-fact status (against REAL), for the Base+VOL+LEV working model `dow10_sty1_c`:**

| # | Stylized fact | Metric used | Value | Target | Verdict |
|:--|:---|:---|---:|---:|:--|
| 1 | Volatility (level) | year VolRatio (raw, vs=1.0) | 0.309 | 1.0 | ✗ FAIL (fixable via vol_scale, §8) |
| 2 | Mean | ~0 both | ≈0 | ≈0 | ✓ PASS (trivial) |
| 3 | Skewness | scipy.skew | -0.078 | -0.954 | ✗ FAIL |
| 4 | Kurtosis | excess kurt | 0.4 | 14.96 | ✗ FAIL |
| 5 | Volatility Clustering | ACF(r²) lag1 | 0.080 | 0.030 | ⚠ PARTIAL (over-clusters) |
| 6 | Leverage Effect | corr(rₜ,|rₜ₊₁|) | -0.023 (old) / +0.01~+0.06 (clean P0/P1) | -0.029 | ⚠ UNRELIABLE — sign flips across runs; within sampling noise at 252d (see p0p1_report §8) |
| 7 | Heavy Tail | Hill α | 5.44 | 2.79 | ✗ FAIL (too light) |
| 8 | Tail Ratio (q95/q05) | P_syn/P_real | ≈0.21 / ≈0.17 | 1.0 | ✗ FAIL (from `diagnostic_dow10_sty1_c`) |
| 9 | Hill Tail Exponent | = #7 | 5.44 | 2.79 | ✗ FAIL |
| 10 | Jump / Extreme | BNS RJ | 0.012 | 0.249 | ✗ FAIL |

Source: this-session evaluation of `outputs/dow10_sty1_c/…returns_2024.csv`;
tail-ratio from `outputs/diagnostic_dow10_sty1_c/diagnostic_report.txt` [Q3].

**Note on the "0.557 vol ratio" in prior notes:** that number is the diagnostic's
per-ticker vol_ratio on the **vol_scale=2.0** generation (`diagnostic_dow10_sty1_c` [Q1]
Avg Vol Ratio=0.557 ≈ our vs=2.0 measurement 0.541). The **raw (vs=1.0)** model VolRatio is
**0.309**. These are different measurements of the same model at different vol_scale → do not
conflate.

---

## 5. Cross-Asset Results

Source: `outputs/diagnostic_dow10_sty1_c/cross_asset_diagnostics.csv` + `diagnostic_report.txt` [Q7].
Model: `dow10_sty1_c` (Base+VOL+LEV).

| Metric | Value | Verdict |
|:---|---:|:--|
| Return Correlation MAE | **0.0877** | ✓ PASS |
| Return Correlation RMSE | 0.1168 | — |
| Return Correlation Frobenius | **1.1677** | ✓ (diagnostic: "Preserves Cross-Asset Correlation: YES") |
| AbsReturn Corr MAE | 0.0814 | ✓ |
| SqReturn Corr MAE | 0.0889 | ✓ |
| Correlation matrices (real/syn/diff) | plots `08a/08b/08c` | present |
| PCA eigenvalue spectrum | plot `10` + report | present |
| Tail Dependence | — | ✗ MISSING (not computed) |

**Evaluation-bug check (user's `reshape(B*D,T)` concern): NO BUG FOUND.**
- In the **loss** (`stylized_loss.py`), `reshape(B*n_tickers, T)` keeps each row as one
  ticker's one window; `_differentiable_acf` / leverage operate **per-row along time** and
  average — tickers are NOT mixed within a series. ✓ verified.
- In the **cross-asset evaluation** (`diagnostic_part1.calculate_cross_asset_dependency`),
  inputs are `(T, N_tickers)` matrices; correlation is across ticker columns — correct. ✓
- → Cross-asset results above are **VALID** (not corrupted by flattening).

---

## 6. Oversampling Ablation (negative baseline)

Extreme-window oversampling (train on more extreme windows). Config from `run_dow10_upsample.sh`
(ratio×, threshold q). Re-evaluated this session (consistent protocol).

| Config | Hill α | Kurt | Jump RJ | Vol-ACF | Leverage | Verdict |
|:---|---:|---:|---:|---:|---:|:--|
| up3_q90 | 6.87 | 0.2 | 0.002 | 0.105 | +0.093 | ✗ no tail gain; over-clusters; leverage sign flip |
| up3_q95 | 5.85 | 0.2 | 0.000 | 0.141 | -0.005 | ✗ no tail gain |
| up5_q95 | 7.93 | 0.1 | 0.004 | 0.199 | +0.147 | ✗ worse tail; strong vol-cluster inflation |

**Verdict: Oversampling is INEFFECTIVE for heavy tails / kurtosis / jumps** (all Kurt ≈ 0.1–0.2
vs target 15; Hill unchanged/worse). It distorts vol clustering (0.10–0.20 vs real 0.03) and
leverage sign. Confirmed as a **negative baseline**. Evidence: `outputs/dow10_up*/…returns_2024.csv`.

---

## 7. Loss Ablation & Tail-Preservation Analysis

### 7.1 Core ablation (Baseline → +VOL → +VOL+LEV → +VOL+LEV+TAIL)

| # | Experiment | Status | Evidence |
|:--|:---|:--|:--|
| 1 | **Baseline (MSE only)** | ✗ MISSING (see §3) | checkpoint `dow10` unverified, no eval |
| 2 | Baseline + VOL | ✗ MISSING (isolated) | no VOL-only DOW10 run/eval exists |
| 3 | Baseline + VOL + LEV | ✓ COMPLETED | `dow10_sty1_c` + `diagnostic_dow10_sty1_c` |
| 4 | + VOL + LEV + TAIL | ✓ COMPLETED | `dow10_tail_l03/l10/l30` (§4) |
| 5 | Oversampling (neg. baseline) | ✓ COMPLETED | §6 |

> ⚠️ The ablation ladder is **broken at the bottom**: steps 1 and 2 (pure MSE, and VOL-only)
> were never isolated/evaluated on DOW10. Step 3 is the lowest rung with evidence. Any claim
> of the form "adding VOL/LEV helped vs baseline" is currently **NOT VERIFIED**.

### 7.2 Tail-preservation interventions (all tested this session)

| Intervention | Purpose | Change | Result (vs Base+VOL+LEV) | Success? | Failure evidence |
|:---|:---|:---|:---|:--:|:---|
| Multi-Quantile Pinball | heavier tails | `fat_tail_loss`, Q90/95/99+Q10/05/01, λ→anneal | Hill 5.44→5.94 (λ3.0), Kurt≈0 | ✗ | matches quantile *location*, not shape; §4 |
| GPD/Hill ξ-match (per-ticker) | match tail index | `gpd_tail_loss`, soft-Hill, up/low tails | loss ↓ (0.13→0.076) but Hill 5.44→5.39 | ✗ | shapes conditional-mean, not Gaussian-kernel generation; §4 |
| Student-t noise prior | heavy-tailed generation | `_draw_noise` df=3/5, unit-var normalized | Kurt still ≈0.5 (df3), Hill 5.40 | ✗ | injected heavy tail denoised away by contraction; §4 |
| x_t residual skip | fix denoiser contraction | `use_xt_residual` + learnable gate | mechanism restored (slope 0.05→0.5) but gate decays 0.5→0.1 under full MSE training; VolRatio not improved | ⚠ partial | `exp_denoiser_sensitivity.py`; gate value in checkpoint |
| ε-parameterization | avoid x0-MSE shrinkage | `parameterization=noise` | VolRatio 58 (exploded), Kurt -1.4 (clip-boxed) | ✗ | unstable ancestral sampling; §4 |
| vol_scale (inference) | amplitude only | `--vol_scale` residual rescale | VolRatio→1.003 at vs=3.86; tails unchanged | ✓ (amplitude only) | §8 |

**Implementation confirmation (user §12):** the Multi-Quantile Pinball tail loss the user
describes **already exists and is tested** — `models_diffusion/stylized_loss.py:fat_tail_loss`
+ `_DEFAULT_TAIL_QUANTILES = {0.99,0.95,0.90,0.10,0.05,0.01}` with delayed λ annealing
(`tail_anneal_delay`). Do NOT re-implement. Its results are in §4 (ineffective for tails).

---

## 8. λ_tail Ablation

**Requested grid `{0, 0.01, 0.03, 0.05, 0.10}`: `✗ MISSING`.**

What exists instead:
- λ_tail = 0 (= `dow10_sty1_c`, Base+VOL+LEV) ✓
- λ_tail = 0.3, 1.0, 3.0 (this session, save_last, full 100ep) ✓ — §4
- Prior notes mention λ_tail = 0.05 and 0.30 runs, but **best-val checkpoint (variance-shrinking) protocol → `? UNCERTAIN` / not comparable** to the save_last sweep.

**The fine-grained small-λ grid `{0.01, 0.03, 0.05, 0.10}` was never run.** More importantly,
**no λ_tail run measured the trade-off vs Cross-Asset Correlation / Vol-Clustering / Leverage**
in one comparable table — the user's stated motivation. → P1, §15.

Observed so far (coarse): raising λ_tail did not create tails but **did perturb leverage**
(λ0.3 → +0.063, wrong sign; recovers by λ3.0). So a tail↔leverage interaction exists but is
under-characterized.

---

## 9. Visualization Summary

For `dow10_sty1_c` (`outputs/diagnostic_dow10_sty1_c/plots/`), present:

| Plot | File | Present |
|:---|:---|:--:|
| Return histogram + KDE | `01_histogram_kde.png` | ✓ |
| ECDF | `02_ecdf.png` | ✓ |
| QQ plot | `03_qqplot.png` | ✓ |
| Boxplot | `04_boxplot.png` | ✓ |
| Abs-return distribution | `05_abs_return_dist.png` | ✓ |
| CCDF (tail) | `06_ccdf.png` | ✓ |
| ACF of returns | `07_acf_return.png` | ✓ |
| ACF of abs returns | `07_acf_abs_return.png` | ✓ |
| ACF of squared returns | `07_acf_squared_return.png` | ✓ |
| Correlation heatmap real/syn/diff | `08a/08b/08c_corr_*.png` | ✓ |
| Skewness scatter | `09_skewness_scatter.png` | ✓ |
| PCA eigenvalues | `10_pca_eigenvalues.png` | ✓ |
| Price trajectory (visual only) | `outputs/dow10_sty1_c/dow10_2024_price.png` | ✓ |

Rolling-volatility plot: not explicitly present (ACF/dist cover clustering). All plots are for
the Base+VOL+LEV model; other configs have generated CSVs but no full plot set.

---

## 10. Price Comparison Classification

- Real-2024 vs Synthetic price series exist (`dow10_2024_price.png`, `..._generated_prices_2024.csv`).
- Classification: **Visual Inspection only — NOT a point-forecasting accuracy metric.**
- Reconstruction verified **chained** from generated returns (`price[t+1]=price[t]·exp(r̂_t)`,
  generated previous day, no per-window reset) → `generate_dow10.py:259-267`. ✓ correct.

---

## 11. Failure Modes & Root-Cause Hypothesis

### Primary Hypothesis (NOT proven as sole cause)
> Under `x_start` + MSE denoising, the denoiser converges toward the **conditional mean**
> E[x₀|x_t, cond] of near-unpredictable returns → **variance / kurtosis / skew / jump
> shrinkage**. Low-frequency high-amplitude extremes get averaged out.

### Supporting evidence (this session)
- `exp_condition_collapse.py`: same condition, 40 seeds → near-identical output
  (noise_frac ≈ 0.001, seed_corr ≈ 0.999). Reverse sampling injects noise but it is denoised away.
- `exp_denoiser_sensitivity.py`: at low noise (t=1) denoiser slope ≈ 0.05, out_std/x0_std ≈ 0.1
  — a strong contraction map (no x_t identity path).
- All richness injections fail identically: Pinball, GPD, Student-t prior, oversampling
  (§4, §6, §7) — consistent with a contraction erasing extremes at every reverse step.
- Best-val checkpointing **rewards** shrinkage (val = sampling-MSE vs unpredictable returns →
  lower for flatter output); `save_last` needed. Residual gate decays 0.5→0.1 under full MSE.

### Honesty caveats (why "sole root cause" is NOT claimed)
- Not isolated against a **pure-MSE baseline** (§3 missing) — cannot attribute shrinkage purely
  to x_start+MSE vs the VOL/LEV losses also present in every evaluated model.
- Under-training is only partially ruled out (res_full trained 100ep and still collapsed, which
  argues against under-training, but other configs' epoch counts are `?`).
- ε-parameterization was expected to reduce x0-mean-shrinkage but instead **exploded** — so the
  mechanism is not cleanly "x0 vs ε"; sampler stability confounds it.

→ Label: **Primary Hypothesis, strongly supported, not proven as the unique cause.**

---

## 12. Missing / Invalid experiments (summary)

| Item | Status | Note |
|:---|:--|:---|
| Pure Diffusion+MSE baseline (evaluated) | ✓ DONE (P0) | `dow10_abl_base`; see `p0p1_report.md` |
| VOL-only ablation | ✓ DONE (P0) | `dow10_abl_vol` |
| 4-rung ablation on one protocol | ✓ DONE (P0) | `p0p1_report.md` |
| λ_tail {0,0.05,0.1,0.3,1.0} + trade-off table | ✓ DONE (P1) | `p0p1_report.md` |
| Tail Dependence (cross-asset) | ✓ ADDED (P1) | `eval_p0p1.py:lower_tail_dep` |
| Downstream RL / Portfolio (Exp 4) | ✗ MISSING (P2) | no code/output found at all — still 0 experiments |
| `dow10` / `dow10_sty1` / `dow10_sty0.3` configs | ? UNCERTAIN | superseded by clean P0 runs |
| Prior λ_tail 0.05/0.30, old 5-ticker diagnostics | NOT DIRECTLY COMPARABLE | best-val / different dataset |
| reshape(B*D,T) cross-asset bug | NOT FOUND (valid) | §5 |

---

## 13. Recommended Next Experiments (priority)

See EXECUTIVE_SUMMARY.md §10 for the top-3. Ranked list in §15 below.

---

## 14. Overall Research Findings

1. The DOW10 diffusion model **learns cross-asset correlation structure well** (Corr MAE 0.088,
   "YES") and **leverage sign** (−0.023 vs −0.029), and reconstructs plausible price paths.
2. It **fails on the tail/extreme family**: heavy tails (Hill 5.4 vs 2.8), kurtosis (0.4 vs 15),
   negative skew (−0.08 vs −0.95), jumps (RJ 0.01 vs 0.25) — all under-generated.
3. Volatility **level** is shrunk (VolRatio 0.31) but **fully correctable** by inference
   `vol_scale` (=3.86 → 1.003); this does not touch tail shape.
4. **Every tail intervention tried failed for one unifying reason** — the denoiser is a
   contraction map that erases injected/pushed extremes; loss-space (Pinball, GPD) and
   input-space (Student-t) fixes are both defeated. Oversampling likewise ineffective.
5. The reliable comparisons are internally consistent (same protocol); the older/cross-dataset
   numbers are not comparable and must not be mixed in.

---

## 15. Limitations

- **No pure-MSE baseline** → ablation increments are not yet attributable.
- Loss configs of the 3 oldest tags are unverifiable.
- Heavy tails / kurtosis / skew / jumps are an **accepted limitation** of this
  architecture+objective (see §11); only amplitude is corrected (vol_scale).
- No downstream Portfolio/RL utility evidence exists → the top-level research question
  ("can synthetic data support downstream RL/PO?") is **entirely unanswered**.

---

### Priority list of missing experiments

**P0 (must):**
1. **Pure-MSE DOW10 baseline** — train `--use_volatility_loss False --use_leverage_loss False`,
   save_last, full generate + `diagnostic_analysis.py`. Cost ≈ 1 train (~4 min) + eval.
   Answers: what does MSE alone learn; anchors the whole ablation ladder.
2. **Clean 4-rung ablation on ONE protocol** (Baseline / +VOL / +VOL+LEV / +VOL+LEV+TAIL), all
   save_last + full epochs + vol_scale-calibrated, one comparison table. Cost ≈ 3 trains.
   Answers: marginal contribution of each loss (currently unverified).

**P1 (recommended):**
3. **λ_tail trade-off ablation** `{0, 0.05, 0.1, 0.3, 1.0}` with a single table reporting
   {Hill, Jump, Skew, VolACF, Leverage, **Cross-Asset Corr MAE**}. Cost ≈ 4 trains.
   Answers: does tail pressure trade off against correlation/clustering/leverage.
4. **Tail Dependence** metric added to cross-asset eval (currently missing). Eval-only.

**P2 (optional / bigger):**
5. **Downstream RL/Portfolio utility** — train an RL/MVO allocator on synthetic vs real, test
   both on the **same real 2024 test set**. This is the actual thesis question; large effort,
   needs a defined allocator + backtest harness (none exists yet).
6. If heavy tails must be *generated* (not accepted as limitation): a non-MSE objective
   (adversarial/score-based) or post-hoc EVT quantile remap (post-processing, must be disclosed).
