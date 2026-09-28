"""
stylized_loss.py
================
Differentiable Financial Stylized Facts Loss functions.

Three structural constraints:
  1. Volatility Clustering Loss  (L_vol)   — ACF of squared returns
  2. Leverage Effect Loss        (L_lev)   — Corr(r_t, sigma_{t+1})
  3. Fat Tail Loss               (L_tail)  — Multi-Quantile Pinball Loss

All are:
  - Fully differentiable (no numpy, no detach, no .item() in graph)
  - GPU-compatible
  - Batch-aware
  - Numerically stable (epsilon guards)

Controlled by flags:
  args.use_volatility_loss  (bool)
  args.use_leverage_loss    (bool)
  args.use_fat_tail_loss    (bool)
  args.anneal_stylized_loss (bool)

Hyperparameters:
  args.lambda_vol_max    (float, e.g. 0.1)
  args.lambda_lev_max    (float, e.g. 0.1)
  args.lambda_tail_max   (float, e.g. 0.05)
  args.anneal_ratio      (float, e.g. 0.2)
  args.tail_anneal_ratio (float, e.g. 0.4)  — Tail Loss starts later than vol/lev
  args.stylized_max_lag  (int,   e.g. 10)
  args.leverage_window   (int,   e.g. 5)
  args.stylized_timestep_ratio (float, e.g. 0.5)
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _differentiable_acf(x: torch.Tensor, lag: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute ACF at a given lag for a 1-D or 2-D tensor (differentiable).

    Args:
        x   : (T,) or (B, T)  — the time series
        lag : positive integer
        eps : small constant for numerical stability

    Returns:
        scalar tensor  — mean ACF over the batch dimension (if 2-D)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)          # (1, T)

    T = x.shape[-1]
    if lag >= T:
        return x.new_zeros(1).squeeze()

    x_lead = x[:, lag:]             # (B, T-lag)
    x_lag  = x[:, :T - lag]        # (B, T-lag)

    # demean each series independently
    mu_lead = x_lead.mean(dim=-1, keepdim=True)
    mu_lag  = x_lag.mean(dim=-1, keepdim=True)

    x_lead_c = x_lead - mu_lead
    x_lag_c  = x_lag  - mu_lag

    # numerator: covariance
    cov = (x_lead_c * x_lag_c).mean(dim=-1)          # (B,)

    # denominator: product of stds
    std_lead = x_lead_c.pow(2).mean(dim=-1).add(eps).sqrt()  # (B,)
    std_lag  = x_lag_c.pow(2).mean(dim=-1).add(eps).sqrt()   # (B,)

    acf = cov / (std_lead * std_lag + eps)            # (B,)
    return acf.mean()                                  # scalar


def _differentiable_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Pearson correlation between two series (differentiable).

    Args:
        x, y : (B, T) tensors
        eps  : numerical stability guard

    Returns:
        scalar — mean correlation over batch
    """
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)

    cov  = (x_c * y_c).mean(dim=-1)                        # (B,)
    std_x = x_c.pow(2).mean(dim=-1).add(eps).sqrt()        # (B,)
    std_y = y_c.pow(2).mean(dim=-1).add(eps).sqrt()        # (B,)

    corr = cov / (std_x * std_y + eps)                     # (B,)
    return corr.mean()                                       # scalar


# ─────────────────────────────────────────────────────────────
# Core Loss Functions
# ─────────────────────────────────────────────────────────────

def volatility_clustering_loss(
    r_gen:  torch.Tensor,
    r_real: torch.Tensor,
    max_lag: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Volatility Clustering Loss.

    Matches the ACF of squared returns between generated and real series:
        L_vol = sum_{k=1}^{max_lag}  (ACF_k(r_gen^2) - ACF_k(r_real^2))^2

    Args:
        r_gen  : (B, T) — generated log-returns (already in raw / denormalized space)
        r_real : (B, T) — real log-returns from the same batch window
        max_lag: K, number of ACF lags to match
        eps    : numerical stability

    Returns:
        scalar loss tensor (differentiable w.r.t. r_gen)
    """
    r_gen_sq  = r_gen.pow(2)
    r_real_sq = r_real.pow(2)

    total_loss = r_gen.new_zeros(1).squeeze()

    for k in range(1, max_lag + 1):
        acf_gen  = _differentiable_acf(r_gen_sq,  k, eps)
        # real ACF: detach so gradient flows only through generated
        acf_real = _differentiable_acf(r_real_sq, k, eps).detach()
        total_loss = total_loss + (acf_gen - acf_real).pow(2)

    return total_loss / max_lag   # average over lags → stable scale


def leverage_effect_loss(
    r_gen:  torch.Tensor,
    r_real: torch.Tensor,
    leverage_window: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Leverage Effect Loss.

    Matches the correlation between returns and future realized volatility:
        sigma_{t+1} = sqrt(mean(r_{t+1:t+H+1}^2))
        L_lev = (Corr(r_gen_t, sigma_gen_{t+1}) - Corr(r_real_t, sigma_real_{t+1}))^2

    Args:
        r_gen  : (B, T) — generated log-returns
        r_real : (B, T) — real log-returns from the same window
        leverage_window: H, number of steps for realized vol estimation
        eps    : numerical stability

    Returns:
        scalar loss tensor (differentiable w.r.t. r_gen)
    """
    H = leverage_window
    T = r_gen.shape[-1]

    if T <= H + 1:
        # Not enough time steps — skip this loss
        return r_gen.new_zeros(1).squeeze()

    T_valid = T - H   # valid length after alignment

    # r_t: returns at time t  (0 .. T_valid-1)
    r_gen_t  = r_gen[:,  :T_valid]     # (B, T_valid)
    r_real_t = r_real[:, :T_valid]     # (B, T_valid)

    # sigma_{t+1}: realized vol over [t+1, t+H]
    # Use a rolling window:  for each t, sigma_{t+1} = sqrt(mean(r[t+1:t+1+H]^2))
    # Stack shifted windows and average
    vol_gen_windows  = torch.stack(
        [r_gen[:,  j+1 : j+1+T_valid].pow(2) for j in range(H)], dim=-1
    ).mean(dim=-1).add(eps).sqrt()   # (B, T_valid)

    vol_real_windows = torch.stack(
        [r_real[:, j+1 : j+1+T_valid].pow(2) for j in range(H)], dim=-1
    ).mean(dim=-1).add(eps).sqrt()   # (B, T_valid)

    rho_gen  = _differentiable_corr(r_gen_t,  vol_gen_windows,  eps)
    # real correlation: detach so gradient flows only through generated
    rho_real = _differentiable_corr(r_real_t, vol_real_windows, eps).detach()

    return (rho_gen - rho_real).pow(2)


# ─────────────────────────────────────────────────────────────
# Fat Tail Loss  (Multi-Quantile Pinball Loss)
# ─────────────────────────────────────────────────────────────

# Default quantile configuration.
# Upper tail: Q90, Q95, Q99  (model tends to be too thin on the right)
# Lower tail: Q10, Q05, Q01  (model tends to be too thin on the left)
# Weights: more extreme quantile → higher weight, because the diagnostic
# directly measures how many returns exceed Q95/Q05 (Tail Ratio).
#
# Weight rationale:
#   Q90 / Q10 → weight 1.0  (moderate tail)
#   Q95 / Q05 → weight 2.0  (main diagnostic target)
#   Q99 / Q01 → weight 3.0  (extreme events; capped at 3× to avoid
#                             numerical instability from rare samples)
#
# Note: we do NOT use weight 4× for Q99/Q01 because with pred_len=21
# there may be 0 or 1 samples per series beyond Q99, causing very high
# gradient variance.  3× is a stable upper bound.

_DEFAULT_TAIL_QUANTILES = [
    (0.99, 3.0),   # Upper extreme
    (0.95, 2.0),   # Upper main target
    (0.90, 1.0),   # Upper moderate
    (0.10, 1.0),   # Lower moderate
    (0.05, 2.0),   # Lower main target
    (0.01, 3.0),   # Lower extreme
]


def _pinball_loss(
    r_gen:    torch.Tensor,   # (N,) or (B, T) — flattened generated returns
    quantile: float,          # target quantile in (0, 1)
    q_real:   torch.Tensor,   # scalar — the real quantile value (detached)
) -> torch.Tensor:
    """
    Pinball (Quantile Regression) Loss at a single quantile.

    For a target quantile τ ∈ (0, 1) and threshold q (from real data):

        L_pinball(τ) = τ·mean(max(r_gen − q, 0))
                     + (1−τ)·mean(max(q − r_gen, 0))

    This is equivalent to the standard asymmetric absolute deviation loss
    used in quantile regression.  Every generated sample contributes a
    non-zero gradient, avoiding the sparse-gradient problem of direct
    MSE-on-quantile approaches.

    Mathematical interpretation:
      - If r_gen > q: penalise by τ × overshoot (small τ → small upper penalty)
      - If r_gen < q: penalise by (1−τ) × undershoot
      → Minimising this loss shifts the τ-quantile of r_gen toward q.

    For upper tail (τ = 0.95):  (1−τ) = 0.05 is small, so the loss naturally
    allows most samples to stay below q95 but penalises all above with 0.95.
    This creates a strong gradient pushing the upper tail of r_gen toward q95_real.

    Args:
        r_gen    : flattened generated returns  (N,)
        quantile : τ
        q_real   : real quantile value (detached from graph)

    Returns:
        scalar loss tensor (differentiable w.r.t. r_gen)
    """
    tau = quantile
    diff = r_gen - q_real
    upper = F.relu(diff)           # max(r_gen - q, 0)  — overshoot
    lower = F.relu(-diff)          # max(q - r_gen, 0)  — undershoot
    return tau * upper.mean() + (1.0 - tau) * lower.mean()


def fat_tail_loss(
    r_gen:  torch.Tensor,   # (B, T) — generated log-returns
    r_real: torch.Tensor,   # (B, T) — real log-returns (same batch window)
    quantiles: list = None, # list of (quantile, weight) pairs; None → default
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Multi-Quantile Pinball Tail Loss.

    Simultaneously constrains both upper and lower tails by matching
    multiple quantiles of generated returns to the corresponding real
    quantiles.  More extreme quantiles receive higher weights to focus
    the gradient signal where the diagnostic deficiency is largest.

    Loss formula:
        L_tail = Σ_q  w_q × PinballLoss(τ_q, q_real)

    Default quantile set and weights:
        Upper  Q99 (w=3) · Q95 (w=2) · Q90 (w=1)
        Lower  Q01 (w=3) · Q05 (w=2) · Q10 (w=1)
        ──── total weight normaliser = sum(weights) = 12

    Why Pinball and not MSE-on-quantile?
        Direct MSE: gradient is non-zero ONLY at the ~1 sample near the
        quantile boundary (sort-based quantile has a sparse gradient).
        Pinball: gradient is non-zero for EVERY generated sample (those
        above the threshold push right-tail up; those below pull it up).
        With pred_len=21, batch_size=32: Pinball has 32×21=672 gradient
        contributions per series; MSE-on-quantile has ≈1.

    Why not Hill or GPD?
        Both require stable tail-sample estimation.  With ≈67 tail samples
        per batch (top 10%), Hill α and GPD MoM have very high variance,
        leading to gradient instability.  Pinball uses ALL samples.

    Interaction with other losses:
        - Volatility: Pinball increases tail variance → may slightly increase
          r² ACF signal.  Direction is cooperative, not competitive.
        - Leverage: Pinball is symmetric (both tails), so it does not
          directly impose skew or leverage. Risk is low.
        - Cross-Asset Correlation: Pinball acts independently per series
          (no cross-ticker constraint), so correlation structure is
          preserved through the MSE path.  Marginal risk only.

    Args:
        r_gen     : (B, T) generated returns  — gradient flows through this
        r_real    : (B, T) real returns       — used only to compute q_real (detached)
        quantiles : list of (τ, weight) pairs; None → _DEFAULT_TAIL_QUANTILES
        eps       : numerical guard

    Returns:
        scalar loss tensor (differentiable w.r.t. r_gen)
    """
    if quantiles is None:
        quantiles = _DEFAULT_TAIL_QUANTILES

    # Flatten to 1-D for quantile computation:
    #   r_gen_flat  (B*T,) — generated   (gradient flows)
    #   r_real_flat (B*T,) — real        (detached reference)
    r_gen_flat  = r_gen.reshape(-1)           # gradient flows
    r_real_flat = r_real.reshape(-1).detach() # reference only

    total_loss   = r_gen.new_zeros(1).squeeze()
    total_weight = 0.0

    for tau, weight in quantiles:
        # Real quantile: computed from the batch's real returns (detached)
        q_real = torch.quantile(r_real_flat, tau)

        pl = _pinball_loss(r_gen_flat, tau, q_real)
        total_loss   = total_loss + weight * pl
        total_weight += weight

    # Normalise by total weight so the loss scale is independent of the
    # number of quantiles used (similar to how vol_loss divides by max_lag)
    return total_loss / (total_weight + eps)


# ─────────────────────────────────────────────────────────────
# GPD / Hill Tail-Index Matching Loss  (per-ticker)
# ─────────────────────────────────────────────────────────────

def _soft_hill_xi(r, q=0.90, tau=0.10, eps=1e-6):
    """
    可微的 soft-Hill 尾指數 ξ（上尾）。

        u  = q-分位數（detached，作為 POT 門檻，q 高 → u>0）
        ξ  = Σ_i w_i · log1p(relu(r_i − u)/u)  /  Σ_i w_i
        w_i = sigmoid((r_i − u)/tau)            # soft membership，梯度不只流進硬門檻樣本

    log1p(relu(·)/u) 在門檻下為 0、門檻上等於 Hill 的 log-excess log(r_i/u)，
    對 r 可微且恆有定義（u>0）。ξ 即 GPD shape / Pareto 尾指數（tail index α = 1/ξ）。
    尺度不變（門檻隨序列縮放），故不需額外標準化。
    """
    u = torch.quantile(r.detach(), q)
    u = torch.clamp(u, min=eps)
    excess = F.relu(r - u)
    logexc = torch.log1p(excess / u)
    w = torch.sigmoid((r - u) / (tau + eps))
    return (w * logexc).sum() / (w.sum() + eps)


def gpd_tail_loss(close_gen, close_real, q=0.90, tau=0.10):
    """
    Per-ticker GPD/Hill 尾指數匹配損失（上下尾分開）。

        L_GPD = mean_j ( |ξ_gen_up_j − ξ_real_up_j| + |ξ_gen_lo_j − ξ_real_lo_j| )

    close_gen / close_real: (B, K, T)  K=n_tickers（僅 rclose 通道）。
    每檔用整個 batch×time 估 ξ（尾部樣本更多、更穩），再平均 K 檔。
    下尾 = 對 −r 取上尾（順帶對負偏有幫助）。real 端 detached。
    """
    B, K, T = close_gen.shape
    total = close_gen.new_zeros(1).squeeze()
    for j in range(K):
        g  = close_gen[:, j, :].reshape(-1)
        rl = close_real[:, j, :].reshape(-1).detach()
        xi_g_up = _soft_hill_xi(g,  q, tau)
        xi_r_up = _soft_hill_xi(rl, q, tau)
        xi_g_lo = _soft_hill_xi(-g,  q, tau)
        xi_r_lo = _soft_hill_xi(-rl, q, tau)
        total = total + (xi_g_up - xi_r_up).abs() + (xi_g_lo - xi_r_lo).abs()
    return total / max(K, 1)


def anneal_linear(progress: float, max_weight: float, anneal_ratio: float) -> float:
    """
    Linear warm-up annealing.

    progress = current_step / total_steps  (0 → 1)

    weight = max_weight * min(progress / anneal_ratio, 1.0)

    Rises linearly from 0 to max_weight over the first anneal_ratio fraction
    of training, then stays at max_weight.

    Recommended for DDPM: simple and predictable; loss scale grows at a
    constant rate so the optimizer is never shocked by a sudden jump.
    """
    if anneal_ratio <= 0:
        return max_weight
    ratio = min(progress / anneal_ratio, 1.0)
    return max_weight * ratio


def anneal_sigmoid(progress: float, max_weight: float, anneal_ratio: float,
                   steepness: float = 10.0) -> float:
    """
    Smooth sigmoid-like annealing.

    Rises smoothly from ~0 to max_weight, with the midpoint at
    anneal_ratio / 2 and transition completed by anneal_ratio.

    This is gentler than linear: the rate of increase is slow at both ends
    and fastest in the middle.  Useful when the Stylized Loss is volatile
    at initialization and you want an especially gradual turn-on.

    Formula:
        midpoint = anneal_ratio / 2
        x_shifted = steepness * (progress - midpoint)
        sigma = 1 / (1 + exp(-x_shifted))
        weight = max_weight * sigma / sigma_at_anneal_ratio
    """
    import math
    if anneal_ratio <= 0:
        return max_weight
    midpoint  = anneal_ratio / 2.0
    x_shifted = steepness * (progress - midpoint)
    sigma     = 1.0 / (1.0 + math.exp(-x_shifted))
    # Normalize so that at progress=anneal_ratio the weight is ≈ max_weight
    x_end     = steepness * (anneal_ratio - midpoint)
    sigma_end = 1.0 / (1.0 + math.exp(-x_end))
    normalized = sigma / (sigma_end + 1e-8)
    return max_weight * min(normalized, 1.0)


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

def compute_stylized_loss(
    model_out: torch.Tensor,   # (B, D, T)  — predicted x0, shape: [batch, features, time]
    x_real:    torch.Tensor,   # (B, D, T)  — real x0 from the same batch
    t:         torch.Tensor,   # (B,)       — diffusion timestep index
    num_timesteps: int,
    args,
    current_step: int,
    total_steps:  int,
) -> tuple:
    """
    Compute weighted Stylized Facts Loss terms.

    Returns (vol_loss, lev_loss, tail_loss, gpd_loss,
             lambda_vol, lambda_lev, lambda_tail, lambda_gpd).
    Zero tensors are returned when the corresponding flag is off.

    Note on tensor layout:
      model_out / x_real shape: (B, D, T) — i.e. [batch, num_features, time]
      We reshape to (B*n_tickers, T) so each rclose channel is treated as an
      independent return series.

    Tail Loss annealing strategy:
      lambda_tail uses a DELAYED start compared to vol/lev.  The tail loss is
      only turned on after anneal_ratio * tail_anneal_delay fraction of training,
      allowing the diffusion model to first establish a stable trajectory before
      the tail constraint kicks in.  This prevents early training instability
      when predicted x0 is still very noisy.
    """
    zero = model_out.new_zeros(1).squeeze()

    # ── Step 1: Check if we should compute (timestep gate: Option C) ────
    t_ratio = getattr(args, 'stylized_timestep_ratio', 0.5)
    low_noise_mask = (t.float() < num_timesteps * t_ratio)
    if not low_noise_mask.any():
        return zero, zero, zero, zero, 0.0, 0.0, 0.0, 0.0

    model_out_ln = model_out[low_noise_mask]   # (B', D, T)
    x_real_ln    = x_real[low_noise_mask]      # (B', D, T)

    B, D, T = model_out_ln.shape

    # ── Channel Filtering ───────────────────────────────────────────────
    # In multi-feature financial datasets (e.g., 3 features per ticker: rclose, rhigh, rlow),
    # Stylized Facts (ACF of squared returns, Leverage Effect) are mathematically defined
    # ONLY on signed cross-day returns (rclose).
    # Filtering for rclose (channels 0, 3, 6, ...) prevents gradient mixing and
    # leverage effect corruption caused by intraday high/low magnitude channels.
    if D % 3 == 0:
        n_tickers = D // 3
        rclose_idx = [i * 3 for i in range(n_tickers)]
        model_out_close = model_out_ln[:, rclose_idx, :]   # (B, n_tickers, T)
        x_real_close    = x_real_ln[:, rclose_idx, :]       # (B, n_tickers, T)
        r_gen  = model_out_close.reshape(B * n_tickers, T)
        r_real = x_real_close.reshape(B * n_tickers, T)
    else:
        # Fallback for single-feature datasets where all channels are returns
        model_out_close = model_out_ln                      # (B, D, T) — 每通道視為一檔
        x_real_close    = x_real_ln
        r_gen  = model_out_ln.reshape(B * D, T) #如果不是三維，就是其他欄位，就Flatten
        r_real = x_real_ln.reshape(B * D, T)

    # ── Step 2: Compute annealing weights ───────────────────────────────
    progress     = current_step / max(total_steps, 1)
    anneal_ratio = getattr(args, 'anneal_ratio', 0.2)
    use_linear   = getattr(args, 'anneal_linear', True)

    lambda_vol_max  = getattr(args, 'lambda_vol_max',  0.1)
    lambda_lev_max  = getattr(args, 'lambda_lev_max',  0.1)
    lambda_tail_max = getattr(args, 'lambda_tail_max', 0.05)
    lambda_gpd_max  = getattr(args, 'lambda_gpd_max',  0.0)

    # Tail Loss uses a delayed annealing schedule:
    # It starts ramping up only after `tail_anneal_delay` fraction of training,
    # giving the diffusion model time to stabilize before the tail constraint is applied.
    # Example: anneal_ratio=0.2, tail_anneal_delay=2.0
    #   → vol/lev reach max at progress=0.2 (20% of training)
    #   → tail starts ramping at progress=0.2, reaches max at progress=0.4 (40%)
    tail_anneal_delay = getattr(args, 'tail_anneal_delay', 2.0)  # multiplier on anneal_ratio
    tail_anneal_ratio = anneal_ratio * tail_anneal_delay

    if getattr(args, 'anneal_stylized_loss', True):
        if use_linear:
            lambda_vol  = anneal_linear(progress, lambda_vol_max,  anneal_ratio)
            lambda_lev  = anneal_linear(progress, lambda_lev_max,  anneal_ratio)
            lambda_tail = anneal_linear(progress, lambda_tail_max, tail_anneal_ratio)
            lambda_gpd  = anneal_linear(progress, lambda_gpd_max,  tail_anneal_ratio)
        else:
            lambda_vol  = anneal_sigmoid(progress, lambda_vol_max,  anneal_ratio)
            lambda_lev  = anneal_sigmoid(progress, lambda_lev_max,  anneal_ratio)
            lambda_tail = anneal_sigmoid(progress, lambda_tail_max, tail_anneal_ratio)
            lambda_gpd  = anneal_sigmoid(progress, lambda_gpd_max,  tail_anneal_ratio)
    else:
        lambda_vol  = lambda_vol_max
        lambda_lev  = lambda_lev_max
        lambda_tail = lambda_tail_max
        lambda_gpd  = lambda_gpd_max

    # ── Step 3: Compute Volatility Clustering Loss ───────────────────────
    vol_loss = zero
    if getattr(args, 'use_volatility_loss', False) and lambda_vol > 0:
        max_lag  = getattr(args, 'stylized_max_lag', 10)
        vol_loss = volatility_clustering_loss(r_gen, r_real, max_lag=max_lag)

    # ── Step 4: Compute Leverage Effect Loss ─────────────────────────────
    lev_loss = zero
    if getattr(args, 'use_leverage_loss', False) and lambda_lev > 0:
        lev_window = getattr(args, 'leverage_window', 5)
        lev_loss   = leverage_effect_loss(r_gen, r_real, leverage_window=lev_window)

    # ── Step 5: Compute Fat Tail Loss (Multi-Quantile Pinball) ───────────
    tail_loss = zero
    if getattr(args, 'use_fat_tail_loss', False) and lambda_tail > 0:
        tail_loss = fat_tail_loss(r_gen, r_real)

    # ── Step 6: Compute GPD/Hill Tail-Index Matching Loss (per-ticker) ───
    gpd_loss = zero
    if getattr(args, 'use_gpd_tail_loss', False) and lambda_gpd > 0:
        gpd_q   = getattr(args, 'gpd_threshold_q', 0.90)
        gpd_tau = getattr(args, 'gpd_soft_tau',    0.10)
        gpd_loss = gpd_tail_loss(model_out_close, x_real_close, q=gpd_q, tau=gpd_tau)

    return (vol_loss, lev_loss, tail_loss, gpd_loss,
            lambda_vol, lambda_lev, lambda_tail, lambda_gpd)


# ===========================================================
# Energy Score（獎勵「分散度」的 proper scoring rule）
# ===========================================================
def energy_score(samples: torch.Tensor, target: torch.Tensor, beta: float = 1.0):
    """多變量 energy score（越小越好）。

        ES = (1/m) Σ_i ||X_i − y||^β  −  1/(2m(m−1)) Σ_{i≠j} ||X_i − X_j||^β

    第一項是準確度，第二項**獎勵樣本之間的散開程度**（負號）。
    它是嚴格 proper 的 scoring rule：唯有預測分布等於真實條件分布時才最小。

    這正是對付 variance collapse 的工具 —— MSE 的最小值在條件期望，
    對 under-dispersion 沒有任何懲罰；energy score 有。

    samples: (m, B, D, T)   同一個 condition / t 下的 m 個 x0 估計
    target : (B, D, T)      真實 x0
    """
    m = samples.shape[0]
    if m < 2:
        raise ValueError("energy score 需要至少 2 個樣本")
    flat = samples.reshape(m, samples.shape[1], -1)          # (m, B, D*T)
    y = target.reshape(target.shape[0], -1).unsqueeze(0)     # (1, B, D*T)

    # 準確度項
    acc = torch.norm(flat - y, dim=-1).pow(beta).mean(dim=0)          # (B,)

    # 分散度項：所有 i<j 配對
    iu = torch.triu_indices(m, m, offset=1, device=samples.device)
    div = torch.norm(flat[iu[0]] - flat[iu[1]], dim=-1).pow(beta).mean(dim=0)   # (B,)

    return (acc - 0.5 * div).mean()


def energy_score_weight(args, current_step, total_steps):
    """energy score 的權重排程（與其他 stylized loss 共用 annealing 慣例）。"""
    lam_max = float(getattr(args, "lambda_energy_max", 0.0))
    if lam_max <= 0:
        return 0.0
    if not getattr(args, "anneal_stylized_loss", True):
        return lam_max
    progress = (current_step / max(total_steps, 1)) if total_steps else 1.0
    ratio = float(getattr(args, "anneal_ratio", 0.2))
    return anneal_linear(progress, lam_max, ratio)


# ===========================================================
# Cross-Asset Dependency Loss
# ===========================================================
def _corr_matrix_differentiable(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """可微的 Pearson 相關矩陣。

    X: (n_assets, N)  —— 列 = 資產（變數），行 = 觀測值
    回傳 (n_assets, n_assets)

    刻意寫成「資產為變數、(batch × time) 為觀測值」的形式：
    相關結構是資產之間的性質，不能把資產維度 flatten 進 batch，
    否則算出來的是「同一支股票跨時間」的自相關而不是跨資產相關。
    """
    Xc = X - X.mean(dim=1, keepdim=True)
    N = X.shape[1]
    cov = Xc @ Xc.transpose(0, 1) / max(N - 1, 1)
    std = cov.diagonal().clamp_min(eps).sqrt()
    return cov / (std.unsqueeze(1) * std.unsqueeze(0) + eps)


def cross_asset_corr_loss(model_out: torch.Tensor,
                          x_real: torch.Tensor,
                          t: torch.Tensor,
                          num_timesteps: int,
                          args) -> tuple:
    """L_cross = || Corr(r_gen) - Corr(r_real) ||_F^2

    動機
    ----
    v-parameterization 修好了 stochasticity 與 variance，但擴散注入的噪聲
    ε ~ N(0, I) 在通道之間是**獨立**的。當輸出從「確定性條件均值」變成
    「大部分由噪聲驅動」，跨資產相關就被洗掉（實測平均相關 0.363 → -0.003）。
    baseline 之所以相關性看起來好，是因為它退化成確定性映射、10 檔共用同一個
    條件訊號 —— 是**因為壞掉才對的**。這個 loss 讓模型真正去學 joint dependency。

    參數
    ----
    model_out : (B, D, T) 預測的 x0（呼叫端須已依 parameterization 還原）
    x_real    : (B, D, T) **同一個 batch 的真實 x0**（= 擴散 loss 的 target）
    t         : (B,) diffusion timestep
    args      : 需要 lambda_cross_max / cross_timestep_ratio / cross_min_t_ratio /
                anneal_* 等欄位。cross_min_t_ratio 是**下界**（只算高噪聲樣本），
                cross_timestep_ratio 是上界；預設 0.0/1.0 = 全部 t。

    註：generation-aligned 模式下 model_out 來自純噪聲輸入、與 x_real 並非逐樣本配對。
    這是刻意的 —— 相關矩陣是**母體統計量**，不需要配對；重點是讓兩邊在
    「模型看不到 x0」的相同條件下比較。

    回傳 (loss, lambda_cross, diag) —— diag 含 corr_mae 與 frobenius 供記錄
    """
    zero = torch.tensor(0.0, device=model_out.device, dtype=model_out.dtype)
    diag = dict(corr_mae=0.0, frobenius=0.0, n_obs=0)

    lam_max = float(getattr(args, "lambda_cross_max", 0.0))
    if lam_max <= 0:
        return zero, 0.0, diag

    # ── timestep gate ──
    # 預設 1.0（所有 t 都算）。與其他 stylized loss 的 0.5 不同，理由是：
    # 生成鏈的每一步都用 x0_hat，若只在低 t 施加約束，高 t 的步驟仍會產生
    # 不相關的結構，到低 t 才修已經來不及。
    hi_ratio = float(getattr(args, "cross_timestep_ratio", 1.0))
    lo_ratio = float(getattr(args, "cross_min_t_ratio", 0.0))
    mask = (t.float() < num_timesteps * hi_ratio) & (t.float() >= num_timesteps * lo_ratio)
    if not mask.any():
        return zero, 0.0, diag

    gen = model_out[mask]
    real = x_real[mask]
    B, D, T = gen.shape

    # ── 只取 rclose 通道（相關結構定義在跨日報酬上，不含 intraday high/low）──
    if D % 3 == 0:
        n_assets = D // 3
        idx = [i * 3 for i in range(n_assets)]
        gen, real = gen[:, idx, :], real[:, idx, :]
    else:
        n_assets = D

    # (B, n_assets, T) -> (n_assets, B*T)：資產為變數，batch×time 為觀測值
    Xg = gen.permute(1, 0, 2).reshape(n_assets, -1)
    Xr = real.permute(1, 0, 2).reshape(n_assets, -1)
    n_obs = Xg.shape[1]
    if n_obs < max(2 * n_assets, 30):      # 觀測值太少，相關估計不可靠
        return zero, 0.0, diag

    Cg = _corr_matrix_differentiable(Xg)
    Cr = _corr_matrix_differentiable(Xr)

    dif = Cg - Cr
    loss = dif.pow(2).sum()                # Frobenius 平方（對角線貢獻恆為 0）

    lam = lam_max
    if getattr(args, "anneal_stylized_loss", True):
        progress = (getattr(args, "_cross_progress", None))
        if progress is not None:
            lam = anneal_linear(progress, lam_max,
                                float(getattr(args, "anneal_ratio", 0.2)))

    off = ~torch.eye(n_assets, dtype=torch.bool, device=dif.device)
    diag = dict(corr_mae=float(dif[off].abs().mean().detach()),
                frobenius=float(loss.detach().sqrt()),
                n_obs=int(n_obs))
    return loss, lam, diag
