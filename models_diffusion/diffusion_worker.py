

from typing import List, Optional, Tuple, Union
from typing import Any, Dict
from functools import partial
from inspect import isfunction
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules import loss
import numpy as np

from utils.diffusion_utils import *
from models_diffusion.stylized_loss import compute_stylized_loss  # 新增
from models_diffusion.stylized_loss import energy_score, energy_score_weight
from models_diffusion.stylized_loss import cross_asset_corr_loss
from models_diffusion.condition_loss import condition_consistency_losses


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class Diffusion_Worker(nn.Module):
    
    def __init__(self, args, u_net=None, diff_steps=1000):
        super(Diffusion_Worker, self).__init__()

        self.args = args
        self.device = args.device

        self.parameterization = args.parameterization
        assert self.parameterization in ["noise", "x_start", "v"], 'supporting "eps" / "x0" / "v"'

        # ── 噪聲先驗（opt-in）：gaussian（預設）或 heavy-tailed Student-t ──────
        # Student-t 正規化到單位變異數，只換「尾巴形狀」不換「尺度」，
        # 維持 DDPM schedule 的變異數假設（避免像 ε-param 那樣發散）。
        self.noise_dist = getattr(args, 'noise_dist', 'gaussian')
        self.noise_df   = float(getattr(args, 'noise_df', 5.0))
        
        self.diff_train_steps = diff_steps
        self.diff_test_steps = diff_steps
        
        self.beta_start = 1e-4 # 1e4
        self.beta_end = 2e-2
        self.beta_schedule = "cosine"

        self.v_posterior = 0.0
        self.original_elbo_weight = 0.0
        self.l_simple_weight = 1

        self.loss_type = "l2"

        self.set_new_noise_schedule(None, self.beta_schedule, self.diff_train_steps, self.beta_start, self.beta_end)

        self.clip_denoised = True

        self.total_N = len(self.alphas_cumprod)
        self.T = 1.
        self.eps = 1e-5

        self.nn = u_net

        self.lambda_condition = float(getattr(args, 'lambda_condition', 0.0))
        if not math.isfinite(self.lambda_condition) or self.lambda_condition < 0:
            raise ValueError('lambda_condition must be finite and nonnegative')
        if self.lambda_condition:
            if (getattr(args, 'market_condition_dim', 0) != 2
                    or getattr(args, 'use_future_mixup', True)
                    or getattr(args, 'use_window_normalization', False)):
                raise ValueError('Condition loss requires two market conditions, no mixup and no window normalization')
            for key in ('condition_feature_scaler', 'condition_label_scaler'):
                state = getattr(args, key, None)
                if state is None or not np.isfinite(state['mean']).all() or not np.isfinite(state['scale']).all() or np.any(np.asarray(state['scale']) <= 0):
                    raise ValueError(f'Missing or invalid training scaler: {key}')

        # ── Stylized Loss 訓練步數計數器 ──
        # 用於 Annealing 計算 progress = global_step / total_steps
        self.global_step = 0
        self.total_steps  = 1   # 由 train() 設置，預設為 1 避免除以零
        self._stylized_log_counter = 0  # 內部計數器供 log 模擬

    def set_new_noise_schedule(self, given_betas=None, beta_schedule="linear", diff_steps=1000, beta_start=1e-4, beta_end=2e-2
    ):  
        if exists(given_betas):
            betas = given_betas
        else:
            if beta_schedule == "linear":
                betas = np.linspace(beta_start, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - beta_start) / (np.exp(-betas) + 1) + beta_start
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = beta_start
        self.linear_end = beta_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization in ("noise", "v"):
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x_start":
            lvlb_weights = 0.8 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss  

    def _draw_noise(self, shape, device):
        """
        依 self.noise_dist 抽噪聲。
          gaussian : 標準常態
          studentt : Student-t(df)，正規化到單位變異數（只換尾巴形狀，不換尺度）
                     t = Z·sqrt(df/G),  G~chi2(df);  再乘 sqrt((df-2)/df) 令 Var=1
        """
        if self.noise_dist == 'studentt':
            df = self.noise_df
            z = torch.randn(shape, device=device)
            gamma = torch.distributions.Gamma(
                torch.tensor(df / 2.0, device=device),
                torch.tensor(0.5, device=device))
            g = gamma.sample(shape)                       # chi2(df)
            t = z * torch.sqrt(df / (g + 1e-8))
            return t * math.sqrt((df - 2.0) / df)         # 正規化到單位變異數
        return torch.randn(shape, device=device)

    def q_sample(self, x_start, t, noise=None):

        noise = default(noise, lambda: self.scaling_noise * self._draw_noise(x_start.shape, x_start.device))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, x=None, cond_ts=None):

        # Feed normalized inputs ith shape of [bsz, feas, seqlen]
        # both x and cond are two time seires 

        B = np.shape(x)[0]
        t = torch.randint(0, self.num_timesteps, size=[(B+1)//2,]).long().to(self.device)
        t = torch.cat([t, self.num_timesteps-1-t], dim=0)[:B]

        noise = self._draw_noise(x.shape, x.device)
        x_k = self.q_sample(x_start=x, t=t, noise=noise)
        
        model_out, pred_out = self.nn(x_k, t, cond_ts, y_clean=x)

        if self.parameterization == "noise":
            target = noise 
        elif self.parameterization == "x_start": #直接預測乾淨的原始資料 x0 (股價報酬率)
            target = x
        elif self.parameterization == "v":
            # v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0  (Salimans & Ho 2022)
            # 與 eps 一樣內建 x_t 恆等路徑，但數值在高低 t 兩端都穩定
            target = self.get_v(x, noise, t)
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")
        
        # in the submission version, we calculate time first, and then calculate variable
        f_dim = -1 if self.args.features == 'MS' else 0
        loss = self.get_loss(model_out[:,:,f_dim:], target[:,:,f_dim:], mean=False).mean(dim=2).mean(dim=1)
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss = loss_simple + self.original_elbo_weight * loss_vlb

        self.last_loss_components = {'diffusion_loss': loss.detach()}
        if self.lambda_condition:
            # SPX disables future mixup, so this forward has no clean-future
            # conditioning. Reuse it without drawing new noise or changing t.
            x0_hat = self._to_x0(model_out, x_k, t)
            return_loss, volatility_loss = condition_consistency_losses(
                x0_hat, cond_ts['market'], self.args.condition_feature_scaler,
                self.args.condition_label_scaler,
                close_indices=getattr(self.args, 'condition_close_indices', (0,)))
            weighted = self.lambda_condition * (return_loss + volatility_loss)
            loss = loss + weighted
            self.last_loss_components.update(
                return_condition_loss=return_loss.detach(),
                volatility_condition_loss=volatility_loss.detach(),
                weighted_condition_loss=weighted.detach())

        if self.args.ablation_study_case == "w_pred_loss":
            pred_loss = self.get_loss(pred_out[:,:,f_dim:], x[:,:,f_dim:], mean=False).mean(dim=2).mean(dim=1)
            loss = loss + self.args.weight_pred_loss * pred_loss.mean()

        # ── Cross-Asset Dependency Loss ──
        # L_cross = || Corr(r_gen) - Corr(r_real) ||_F^2
        # r_real 就是本 batch 的真實 x0（= 擴散 loss 的 target），沒有引入任何
        # 新資料；batch 由 DatasetWF 從當期訓練窗 [origin-1260, origin) 取樣，
        # run_one_cycle 另有 assert 保證訓練窗結尾 < origin，因此無 look-ahead。
        if float(getattr(self.args, 'lambda_cross_max', 0.0)) > 0:
            self.args._cross_progress = (self.global_step / max(self.total_steps, 1))
            if getattr(self.args, 'cross_gen_aligned', False):
                # ── Generation-Aligned 模式（處理「障礙 C」）──
                # 問題：訓練時 x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps 永遠殘留真實 x0 的
                #       相關結構（t=95 時仍有 6%），使 loss 量到的 corr_mae 只有 ~0.035，
                #       而純噪聲生成時是 ~0.5 —— loss 收不到真正的誤差訊號。
                # 作法：改用**純 N(0,I)** 當輸入（不經 q_sample，零 x0 殘留）、
                #       t 取高噪聲帶，這正是生成鏈第一步的條件。
                # 注意：此時 x0_hat 與 x_real 並非逐樣本配對，這是刻意的 ——
                #       相關矩陣是母體統計量，重點是兩邊處於相同的「看不到 x0」條件。
                lo = int(float(getattr(self.args, 'cross_gen_t_ratio', 0.9))
                         * self.num_timesteps)
                lo = min(max(lo, 1), self.num_timesteps - 1)
                tg = torch.randint(lo, self.num_timesteps, (x.shape[0],),
                                   device=x.device).long()
                xT = self._draw_noise(x.shape, x.device)      # 純噪聲，非 q_sample
                raw = self.nn(xT, tg, cond_ts, y_clean=None)
                if isinstance(raw, (tuple, list)):
                    raw = raw[0]
                x0_c, t_for_loss = self._to_x0(raw, xT, tg), tg
            else:
                x0_c = self._aux_x0_hat(model_out, x_k, t, cond_ts)
                t_for_loss = t
            cross_loss, lambda_cross, cross_diag = cross_asset_corr_loss(
                model_out=x0_c[:, f_dim:, :],
                x_real=x[:, f_dim:, :],
                t=t_for_loss, num_timesteps=self.num_timesteps, args=self.args)
            loss = loss + lambda_cross * cross_loss
            self._last_cross = (float(cross_loss.detach()), lambda_cross, cross_diag)
            self._cross_log_counter = getattr(self, "_cross_log_counter", 0) + 1
            if self._cross_log_counter % 100 == 1:
                print(f"[CrossAsset] step={self.global_step} "
                      f"L_F2={float(cross_loss.detach()):.4f} (λ={lambda_cross:.4f}, "
                      f"w={lambda_cross * float(cross_loss.detach()):.4f})  "
                      f"corr_mae={cross_diag['corr_mae']:.4f}  "
                      f"frob={cross_diag['frobenius']:.4f}  n_obs={cross_diag['n_obs']}")

        # ── Energy Score（對付 variance collapse）──
        # MSE 的最小值在條件期望，對 under-dispersion 完全沒有懲罰；
        # energy score 是 proper scoring rule，第二項直接獎勵樣本之間的散開程度。
        #
        # 只在**高 t**（低訊噪比）套用，理由：
        #   低 t 時 x_t ≈ x0，真實的條件分布 p(x0 | x_t) 本來就是集中的，
        #   在那裡獎勵分散度是錯的（會逼出假的雜訊）。
        #   高 t 時 x_t 幾乎不帶 x0 的資訊，{x0_hat_i} 實質上就是模型的生成分布，
        #   此時 ES 對到的正是我們要匹配的那個分布。
        lam_es = energy_score_weight(self.args, self.global_step, self.total_steps)
        if lam_es > 0:
            m = int(getattr(self.args, 'energy_n_samples', 3))
            min_ratio = float(getattr(self.args, 'energy_min_t_ratio', 0.6))
            hi = t >= int(min_ratio * self.num_timesteps)
            if hi.any() and m >= 2:
                cb = {key: value[hi] for key, value in cond_ts.items()} if isinstance(cond_ts, dict) else cond_ts[hi]
                xb, tb = x[hi], t[hi]
                def _x0hat(out_, xk_, t_):
                    if self.parameterization == "noise":
                        return self.predict_start_from_noise(xk_, t_, out_)
                    if self.parameterization == "v":
                        return self.predict_start_from_v(xk_, t_, out_)
                    return out_
                hats = []
                for _ in range(m):
                    n_i = self._draw_noise(xb.shape, xb.device)
                    xk_i = self.q_sample(x_start=xb, t=tb, noise=n_i)
                    out_i, _ = self.nn(xk_i, tb, cb, y_clean=xb)
                    hats.append(_x0hat(out_i, xk_i, tb))
                es = energy_score(torch.stack(hats, dim=0)[:, :, f_dim:, :],
                                  xb[:, f_dim:, :],
                                  beta=float(getattr(self.args, 'energy_beta', 1.0)))
                loss = loss + lam_es * es
                self._last_energy = (float(es.detach()), lam_es, int(hi.sum()))

        # ── Stylized Facts-aware Loss (Option C: only low-noise timesteps) ──
        # 只有至少一個 flag 開啟時才進入計算
        use_vol  = getattr(self.args, 'use_volatility_loss', False)
        use_lev  = getattr(self.args, 'use_leverage_loss',  False)
        use_tail = getattr(self.args, 'use_fat_tail_loss',  False)
        if use_vol or use_lev or use_tail:
            # Stylized facts 定義在 x0（報酬）上。x_start param 時 model_out 就是 x0；
            # noise param 時 model_out 是預測噪聲 ε，必須先反推 x0，否則會拿 ε 去比報酬統計量。
            x0_hat = self._aux_x0_hat(model_out, x_k, t, cond_ts)
            # model_out 和 x 的 shape：(B, D, T)  [features are dim-1]
            # 要穩 diffusion loss 的 f_dim 切片，保持一致
            out_slice = x0_hat[:, f_dim:, :]     # (B, D', T)  predicted x0
            x_slice   = x[:, f_dim:, :]          # (B, D', T)

            (vol_loss, lev_loss, tail_loss, gpd_loss,
             lambda_vol, lambda_lev, lambda_tail, lambda_gpd) = compute_stylized_loss(
                model_out=out_slice,
                x_real=x_slice,
                t=t,
                num_timesteps=self.num_timesteps,
                args=self.args,
                current_step=self.global_step,
                total_steps=self.total_steps,
            )

            weighted_vol  = lambda_vol  * vol_loss
            weighted_lev  = lambda_lev  * lev_loss
            weighted_tail = lambda_tail * tail_loss
            weighted_gpd  = lambda_gpd  * gpd_loss
            loss = loss + weighted_vol + weighted_lev + weighted_tail + weighted_gpd

            # 詳細 loss log：每 100 global_step 印一次
            self._stylized_log_counter += 1
            if self._stylized_log_counter % 100 == 1:
                base_loss = (loss.item() - weighted_vol.item() - weighted_lev.item()
                             - weighted_tail.item() - weighted_gpd.item())
                print(
                    f"[Stylized] step={self.global_step} "
                    f"diffusion={base_loss:.5f}  "
                    f"vol={vol_loss.item():.5f} (λ={lambda_vol:.4f}, w={weighted_vol.item():.5f})  "
                    f"lev={lev_loss.item():.5f} (λ={lambda_lev:.4f}, w={weighted_lev.item():.5f})  "
                    f"tail={tail_loss.item():.5f} (λ={lambda_tail:.4f}, w={weighted_tail.item():.5f})  "
                    f"gpd={gpd_loss.item():.5f} (λ={lambda_gpd:.4f}, w={weighted_gpd.item():.5f})  "
                    f"total={loss.item():.5f}"
                )

        # 步數勞進（僅在 training 時進行）
        self.global_step += 1
        return loss

    def _to_x0(self, raw, x_k, t):
        """把 nn 的原始輸出依 parameterization 換算回 x0。"""
        if self.parameterization == "noise":
            return self.predict_start_from_noise(x_k, t, raw)
        if self.parameterization == "v":
            return self.predict_start_from_v(x_k, t, raw)
        return raw

    def _aux_x0_hat(self, model_out, x_k, t, cond_ts):
        """輔助 loss（stylized / cross-asset）要用的 x0_hat。

        預設沿用主 forward 的 model_out —— 但那次 forward 帶了 y_clean=x（future
        mixup），denoiser 的輸入平均混入 50% 的真實 x0，因此算出來的統計量是
        **teacher-forced** 的，與推論行為脫節（實測跨資產相關在 t=99 時
        訓練 0.61 / 推論 0.19）。

        aux_no_teacher_forcing=True 時額外做一次 y_clean=None 的 forward，
        讓輔助 loss 對準真正的生成行為。代價是每步多一次 forward。
        """
        if not getattr(self.args, "aux_no_teacher_forcing", False):
            return self._to_x0(model_out, x_k, t)
        raw = self.nn(x_k, t, cond_ts, y_clean=None)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        return self._to_x0(raw, x_k, t)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # ── v-parameterization（Salimans & Ho 2022, "Progressive Distillation"）──
    # v = sqrt(ab)*eps - sqrt(1-ab)*x0      x0 = sqrt(ab)*x_t - sqrt(1-ab)*v
    # 與 eps 一樣把 x_t 的恆等路徑內建進來，且係數自帶正確的 t 依賴，
    # 因此不需要一個「與 t 無關的純量 gate」去做無法兼顧的折衷。
    def get_v(self, x_start, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond_ts, clip_denoised: bool):

        model_out = self.nn(x, t, cond_ts, y_clean=None)
        
        if self.parameterization == "noise":
            # model_out.clamp_(-10., 10.)
            model_out = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "v":
            model_out = self.predict_start_from_v(x, t, model_out)

        x_recon = model_out

        if clip_denoised:
            x_recon.clamp_(-self.args.our_ddpm_clip, self.args.our_ddpm_clip)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond_ts, clip_denoised=True, repeat_noise=False):

        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond_ts=cond_ts, clip_denoised=clip_denoised)

        noise = self._draw_noise(x.shape, device)   # gaussian 或 heavy-tailed Student-t
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def sample(self, x=None, cond_ts=None, store_intermediate_states=False):

        # Feed normalized inputs ith shape of [bsz, feas, seqlen]
        # both x and cond are two time seires 
        
        history = cond_ts['history'] if isinstance(cond_ts, dict) else cond_ts
        b, d, l = np.shape(history)
        shape = (b, d, np.shape(x)[-1])
        timeseries = self._draw_noise(shape, self.device)   # x_T 初始噪聲（同分佈）
        intermediates = [timeseries.permute(0,2,1)] # return bsz, seqlen, fea_dim

        for i in reversed(range(0, self.num_timesteps)):

            timeseries = self.p_sample(timeseries, torch.full((b,), i, device=self.device, dtype=torch.long), 
                cond_ts, clip_denoised=self.clip_denoised)
            if store_intermediate_states:
                intermediates.append(timeseries.permute(0,2,1))

        outs = timeseries.permute(0,2,1) 


        return outs
