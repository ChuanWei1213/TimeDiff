"""Dow 10 conditional TimeDiff: joint 30-channel generation conditioned on the
equal-weight portfolio's (21-day return, annualized volatility).

Same model recipe as the SPX consistency experiment (FiLM conditioning, frozen AR,
no future mixup, condition-consistency loss), with num_vars = 30.

    python3 dow10_cond_experiment.py download --start 2008-10-01 --end 2023-01-01 \
        --data datasets/dow10/prices.csv
    python3 dow10_cond_experiment.py train --data datasets/dow10/prices.csv \
        --target-start 2009-04-01 --train-end 2017-12-31 --val-end 2019-12-31 \
        --lambda-condition 0.1 --out outputs/dow10/conditional/fixed_scenarios/legacy/train
    python3 dow10_cond_experiment.py generate --mode diagnostic \
        --checkpoint outputs/dow10/conditional/fixed_scenarios/legacy/train/model.pt --out outputs/dow10/conditional/fixed_scenarios/legacy/diagnostic
    python3 dow10_cond_experiment.py generate --mode rl \
        --checkpoint outputs/dow10/conditional/fixed_scenarios/legacy/train/model.pt --out outputs/dow10/conditional/fixed_scenarios/legacy/rl_paths
"""
import argparse
import hashlib
import json
import subprocess
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from output_layout import resolve_output_path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "datasets/dow10/prices.csv"
DEFAULT_SCENARIOS = [(-0.10, 0.40), (-0.20, 0.60), (-0.30, 0.80)]


# ───────────────────────── torch-free helpers (unit-testable) ─────────────────────────
def reconstruct_prices(anchor_close, raw_future, close_idx, high_idx, low_idx):
    """anchor_close [n, A]; raw_future [n, 21, 30] raw log features -> close/high/low [n, 21, A].
    High/Low are clipped so that low <= close <= high (same rule as the SPX sampler)."""
    rc, rh, rl = raw_future[..., close_idx], raw_future[..., high_idx], raw_future[..., low_idx]
    close = anchor_close[:, None, :] * np.exp(np.cumsum(rc, axis=1))
    high = close * np.exp(np.maximum(rh, 0.0))
    low = close * np.exp(np.minimum(rl, 0.0))
    return close, high, low, (rh < 0), (rl > 0)


def mean_pairwise_corr(rc):
    """rc [n, T, A] -> [n] mean off-diagonal correlation of daily returns across assets."""
    out = np.full(rc.shape[0], np.nan)
    A = rc.shape[-1]
    mask = ~np.eye(A, dtype=bool)
    for i, x in enumerate(rc):
        if np.all(x.std(axis=0) > 0):
            out[i] = np.corrcoef(x.T)[mask].mean()
    return out


def scenario_report(scenario_id, target, raw_future, close_idx, high_clip, low_clip, anchors,
                    return_tol, vol_tol, stats_fn):
    realized = stats_fn(raw_future[..., close_idx])
    err = np.abs(realized - np.asarray(target))
    hit = (err[:, 0] <= return_tol) & (err[:, 1] <= vol_tol)
    corr = mean_pairwise_corr(raw_future[..., close_idx])
    per_anchor = pd.DataFrame({"anchor": anchors, "R": realized[:, 0], "V": realized[:, 1]})
    spread = per_anchor.groupby("anchor")[["R", "V"]].std(ddof=1)
    return {
        "scenario": int(scenario_id), "target_return": target[0], "target_volatility": target[1],
        "n_paths": int(len(realized)), "hit_rate": float(hit.mean()),
        "return_mae": float(err[:, 0].mean()), "volatility_mae": float(err[:, 1].mean()),
        "realized_return_mean": float(realized[:, 0].mean()),
        "realized_return_q05": float(np.quantile(realized[:, 0], .05)),
        "realized_return_q95": float(np.quantile(realized[:, 0], .95)),
        "realized_vol_mean": float(realized[:, 1].mean()),
        "realized_vol_q05": float(np.quantile(realized[:, 1], .05)),
        "realized_vol_q95": float(np.quantile(realized[:, 1], .95)),
        "mean_pairwise_corr": float(np.nanmean(corr)),
        "max_abs_daily_return": float(np.abs(raw_future[..., close_idx]).max()),
        "rhigh_clip_rate": float(high_clip.mean()), "rlow_clip_rate": float(low_clip.mean()),
        # only meaningful with several paths per anchor (diagnostic mode)
        "within_anchor_return_std": float(spread.R.mean()) if spread.R.notna().any() else None,
        "within_anchor_vol_std": float(spread.V.mean()) if spread.V.notna().any() else None,
    }, realized, hit


def pick_anchors(dates, start, end, mode, per_year=2):
    dates = pd.DatetimeIndex(dates)
    sel = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if mode == "rl":
        return list(sel)
    picks = []
    for year in sorted(set(sel.year)):
        ys = sel[sel.year == year]
        idx = np.linspace(0, len(ys) - 1, per_year + 2)[1:-1].round().astype(int)
        picks += list(ys[idx])
    return picks


# ───────────────────────────────── torch parts ─────────────────────────────────
def model_args(device, channels=64, parameterization="v", sampler="dpm"):
    import torch
    from data_provider.dow10_cond import SEQ_LEN, PRED_LEN, N_VARS
    return SimpleNamespace(
        device=torch.device(device), num_vars=N_VARS, seq_len=SEQ_LEN, label_len=21,
        pred_len=PRED_LEN, diff_steps=100, sampling_steps=20, UNet_Type="CNN",
        type_sampler=sampler, parameterization=parameterization, features="M",
        market_condition_dim=2, vis_ar_part=False, use_future_mixup=False,
        use_window_normalization=False, ddpm_inp_embed=channels,
        ddpm_dim_diff_steps=channels, ddpm_channels_conv=channels,
        ddpm_channels_fusion_I=channels, ddpm_layers_inp=5, ddpm_layers_I=5,
        ddpm_layers_II=5, cond_ddpm_num_layers=5, cond_ddpm_channels_conv=channels,
        ablation_study_F_type="CNN", ablation_study_case="none",
        our_ddpm_clip=100., noise_dist="gaussian", use_xt_residual=False,
        use_volatility_loss=False, use_leverage_loss=False, use_fat_tail_loss=False,
        lambda_energy_max=0., lambda_cross_max=0.,
    )


def decoder(history, future=None):
    import torch
    from data_provider.dow10_cond import PRED_LEN, N_VARS
    if future is None:
        future = torch.zeros(history.shape[0], PRED_LEN, N_VARS, device=history.device)
    return torch.cat([history[:, -21:], future], dim=1)


def validation_score(model, loader, fs, cs, max_batches):
    import torch
    from data_provider.dow10_cond import CLOSE_INDICES, portfolio_statistics
    model.eval()
    scores = []
    devices = [model.device.index or 0] if model.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(1729)
        for history, _, conditions in islice(loader, max_batches):
            history, conditions = history.to(model.device), conditions.to(model.device)
            out = model(history, None, decoder(history), None, sample_times=1,
                        market_condition=conditions)[0][:, 0]
            raw = fs.inverse(out.cpu().numpy())
            actual = cs.transform(portfolio_statistics(raw[..., CLOSE_INDICES]))
            scores.extend(np.square(actual - conditions.cpu().numpy()).mean(axis=1).tolist())
    return float(np.mean(scores))


def train(cli):
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from data_provider.dow10_cond import (COLUMNS, CLOSE_INDICES, PRED_LEN, Scaler, Dow10Windows,
                                          load_features, split_windows, window_conditions)
    from models_diffusion.DDPM import Model

    out = Path(cli.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    _, features = load_features(cli.data)
    splits = split_windows(features, cli.target_start, cli.train_end, cli.val_end)
    # Scalers fit on TRAIN rows only (from the first train history row to train_end).
    first_row = splits["train"][0] - 63
    fs = Scaler.fit(features.iloc[first_row:].loc[:cli.train_end, COLUMNS].to_numpy())
    raw_train_c = window_conditions(features[COLUMNS].to_numpy(), splits["train"])
    cs = Scaler.fit(raw_train_c)
    train_loader = DataLoader(Dow10Windows(features, splits["train"], fs, cs),
                              batch_size=cli.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Dow10Windows(features, splits["validation"], fs, cs),
                            batch_size=cli.batch_size, shuffle=False)

    args = model_args(cli.device, cli.channels)
    args.lambda_condition = cli.lambda_condition
    if cli.lambda_condition:
        args.condition_feature_scaler = fs.state()
        args.condition_label_scaler = cs.state()
        args.condition_close_indices = list(CLOSE_INDICES)
    model = Model(args).to(args.device)

    def forever(loader):
        while True:
            yield from loader
    stream = forever(train_loader)

    opt = torch.optim.Adam(model.dlinear_model.parameters(), lr=cli.lr)
    for _ in tqdm(range(cli.ar_steps), desc="AR pretraining", disable=cli.no_progress):
        h, y, _c = (v.to(args.device) for v in next(stream))
        opt.zero_grad()
        loss = model.pretrain_forward(h, None, decoder(h, y), None)
        loss.backward()
        opt.step()
    model.freeze_ar()
    ar_snapshot = {k: v.detach().clone() for k, v in model.dlinear_model.state_dict().items()}

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cli.lr)
    model.diffusion_worker.total_steps = cli.steps
    records, best, best_state, best_step = [], float("inf"), None, None
    out.mkdir(parents=True, exist_ok=True)
    bar = tqdm(range(1, cli.steps + 1), desc="Diffusion training", disable=cli.no_progress)
    for step in bar:
        model.train()
        h, y, c = (v.to(args.device) for v in next(stream))
        opt.zero_grad()
        loss = model.train_forward(h, None, decoder(h, y), None, market_condition=c)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        opt.step()
        if step == 1 or step % cli.eval_every == 0 or step == cli.steps:
            score = validation_score(model, val_loader, fs, cs, cli.val_batches)
            records.append({"step": step, "train_loss": loss.item(), "validation_condition_mse": score,
                            **{k: float(v) for k, v in model.diffusion_worker.last_loss_components.items()}})
            tqdm.write(str(records[-1]))
            pd.DataFrame(records).to_csv(out / "training.csv", index=False)
            if np.isfinite(score) and score < best:
                best, best_step = score, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No finite validation score")
    if any(not torch.equal(v, ar_snapshot[k]) for k, v in model.dlinear_model.state_dict().items()):
        raise RuntimeError("AR weights changed after freezing")
    saved_args = vars(args).copy()
    saved_args["device"] = str(args.device)
    meta = {
        "format_version": 1, "universe": "dow10", "columns": COLUMNS,
        "close_indices": list(CLOSE_INDICES),
        "condition": "equal-weight daily-rebalanced portfolio: [cum_return_21d, ann_vol_ddof1]",
        "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                      text=True).stdout.strip() or None,
        "args": saved_args, "feature_scaler": fs.state(), "condition_scaler": cs.state(),
        "target_start": cli.target_start, "train_end": cli.train_end, "val_end": cli.val_end,
        "split_windows": {k: len(v) for k, v in splits.items()},
        "data_sha256": hashlib.sha256(Path(cli.data).read_bytes()).hexdigest(),
        "seed": cli.seed, "best_step": best_step, "best_validation_condition_mse": best,
    }
    torch.save({"state_dict": best_state, "metadata": meta}, out / "model.pt")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    pd.DataFrame(raw_train_c, columns=["return", "volatility"]).to_csv(out / "train_conditions.csv", index=False)
    print(f"Saved {out/'model.pt'} (best step {best_step}, val {best:.4f})")


def generate(cli):
    import torch
    from tqdm.auto import tqdm
    from data_provider.dow10_cond import (COLUMNS, CLOSE_INDICES, TICKERS, Scaler, load_features,
                                          portfolio_statistics, wide_prices, SEQ_LEN, split_windows)
    from models_diffusion.DDPM import Model

    out = Path(cli.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=True)
    meta = ckpt["metadata"]
    args = SimpleNamespace(**meta["args"])
    args.device = torch.device(cli.device)
    model = Model(args).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze_ar()
    model.eval()
    fs, cs = Scaler(**meta["feature_scaler"]), Scaler(**meta["condition_scaler"])

    prices, features = load_features(cli.data)
    if pd.Timestamp(cli.anchor_end) > pd.Timestamp(meta["train_end"]):
        raise ValueError("anchor_end must not exceed the generator train_end (history leakage)")
    values = fs.transform(features[COLUMNS].to_numpy())
    close_w = wide_prices(prices)["close"]
    first_ok = features.index[SEQ_LEN - 1]
    anchors = [a for a in pick_anchors(features.index, cli.anchor_start, cli.anchor_end, cli.mode,
                                       cli.per_year) if a >= first_ok]
    n_rep = 1 if cli.mode == "rl" else cli.paths_per_anchor
    scenarios = [tuple(s) for s in (cli.scenario or DEFAULT_SCENARIOS)]
    jobs = [(a, sid) for sid in range(len(scenarios)) for a in anchors for _ in range(n_rep)]

    torch.manual_seed(cli.seed)
    raws = np.empty((len(jobs), 21, len(COLUMNS)), dtype=np.float32)
    with torch.no_grad():
        for b in tqdm(range(0, len(jobs), cli.batch_size), desc="Generating", disable=cli.no_progress):
            chunk = jobs[b:b + cli.batch_size]
            locs = [features.index.get_loc(a) for a, _ in chunk]
            h = torch.tensor(np.stack([values[p - SEQ_LEN + 1:p + 1] for p in locs]), device=args.device)
            c = torch.tensor(cs.transform([scenarios[s] for _, s in chunk]), device=args.device)
            o = model(h, None, decoder(h), None, sample_times=1, market_condition=c)[0][:, 0]
            raws[b:b + len(chunk)] = fs.inverse(o.cpu().numpy())
    if not np.isfinite(raws).all():
        raise RuntimeError("Non-finite generated samples")

    anchor_dates = np.array([a for a, _ in jobs])
    scen = np.array([s for _, s in jobs])
    anchor_close = close_w.loc[anchor_dates].to_numpy(float)
    high_idx = [i + 1 for i in CLOSE_INDICES]
    low_idx = [i + 2 for i in CLOSE_INDICES]
    close, high, low, hclip, lclip = reconstruct_prices(anchor_close, raws, CLOSE_INDICES, high_idx, low_idx)

    reports, realized_all, hit_all = [], np.empty((len(jobs), 2)), np.empty(len(jobs), bool)
    for sid, target in enumerate(scenarios):
        m = scen == sid
        rep, realized, hit = scenario_report(sid, target, raws[m], CLOSE_INDICES, hclip[m], lclip[m],
                                             anchor_dates[m], cli.return_tolerance, cli.vol_tolerance,
                                             portfolio_statistics)
        reports.append(rep)
        realized_all[m], hit_all[m] = realized, hit
    # Real reference: the most volatile training windows (what the generator has seen).
    starts = np.array(split_windows(features, meta["target_start"], meta["train_end"], meta["val_end"])["train"])
    raw_np = features[COLUMNS].to_numpy()
    real_stats = np.array([portfolio_statistics(raw_np[s:s + 21, CLOSE_INDICES]) for s in starts])
    top = np.argsort(real_stats[:, 1])[-50:]
    real_corr = mean_pairwise_corr(np.stack([raw_np[s:s + 21, CLOSE_INDICES] for s in starts[top]]))

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "paths.npz", tickers=np.array(TICKERS),
        anchor_dates=np.array([pd.Timestamp(a).strftime("%Y-%m-%d") for a in anchor_dates]),
        scenario=scen, target=np.array([scenarios[s] for s in scen]),
        close=close.astype(np.float32), high=high.astype(np.float32), low=low.astype(np.float32),
        realized=realized_all, hit=hit_all, raw_features=raws)
    pd.DataFrame(reports).to_csv(out / "summary.csv", index=False)
    (out / "generation.json").write_text(json.dumps({
        "mode": cli.mode, "n_paths": len(jobs), "n_anchors": len(anchors),
        "first_anchor": str(anchors[0].date()), "last_anchor": str(anchors[-1].date()),
        "scenarios": scenarios, "paths_per_anchor_per_scenario": n_rep,
        "generator_train_end": meta["train_end"], "generator_val_end": meta["val_end"],
        "real_reference_top50_vol_windows": {
            "max_return_drop": float(real_stats[:, 0].min()), "max_vol": float(real_stats[:, 1].max()),
            "mean_pairwise_corr": float(np.nanmean(real_corr))},
        "checkpoint_sha256": hashlib.sha256(Path(cli.checkpoint).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(Path(cli.data).read_bytes()).hexdigest(),
        "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
        "note": "target/scenario/realized are generator labels; never put them in the RL state.",
    }, indent=2) + "\n")
    print(pd.DataFrame(reports).to_string(index=False))


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", default="2008-10-01")
    d.add_argument("--end", default="2023-01-01", help="exclusive")
    d.add_argument("--data", type=Path, default=DEFAULT_DATA)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=DEFAULT_DATA)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--target-start", default="2009-04-01")
    t.add_argument("--train-end", default="2017-12-31")
    t.add_argument("--val-end", default="2019-12-31")
    t.add_argument("--ar-steps", type=int, default=500)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--val-batches", type=int, default=4)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--channels", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--lambda-condition", type=float, default=0.1)
    g = sub.add_parser("generate")
    g.add_argument("--data", type=Path, default=DEFAULT_DATA)
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--mode", choices=["rl", "diagnostic"], required=True)
    g.add_argument("--anchor-start", default="2009-04-01")
    g.add_argument("--anchor-end", default="2017-12-31")
    g.add_argument("--per-year", type=int, default=2, help="diagnostic anchors per year")
    g.add_argument("--paths-per-anchor", type=int, default=100, help="diagnostic only")
    g.add_argument("--scenario", type=float, nargs=2, action="append", metavar=("RETURN", "VOL"))
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--return-tolerance", type=float, default=0.02)
    g.add_argument("--vol-tolerance", type=float, default=0.05)
    for q in (t, g):
        q.add_argument("--seed", type=int, default=2024)
        q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        q.add_argument("--no-progress", action="store_true")
    cli = p.parse_args()
    if hasattr(cli, "out"):
        cli.out = resolve_output_path(cli.out)
    if hasattr(cli, "checkpoint"):
        cli.checkpoint = resolve_output_path(cli.checkpoint)
    if cli.cmd == "download":
        from data_provider.dow10_cond import download_dow10
        download_dow10(cli.data, cli.start, cli.end)
    elif cli.cmd == "train":
        train(cli)
    else:
        generate(cli)


if __name__ == "__main__":
    main()
