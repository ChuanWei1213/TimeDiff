"""Single-index TimeDiff: download, freeze-AR training, conditional sampling."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_provider.spx import (SEQ_LEN, PRED_LEN, FEATURES, Scaler, SPXWindows,
                               download_spx, load_features, market_statistics, split_windows)
from models_diffusion.DDPM import Model

ROOT = Path(__file__).resolve().parent


def model_args(device='cpu', channels=64, conditioned=True, parameterization='v', sampler='dpm'):
    return SimpleNamespace(
        device=torch.device(device), num_vars=3, seq_len=SEQ_LEN, label_len=21,
        pred_len=PRED_LEN, diff_steps=100, sampling_steps=20, UNet_Type='CNN',
        type_sampler=sampler, parameterization=parameterization, features='M',
        market_condition_dim=2 if conditioned else 0, vis_ar_part=False,
        use_window_normalization=False, ddpm_inp_embed=channels,
        ddpm_dim_diff_steps=channels, ddpm_channels_conv=channels,
        ddpm_channels_fusion_I=channels, ddpm_layers_inp=5, ddpm_layers_I=5,
        ddpm_layers_II=5, cond_ddpm_num_layers=5, cond_ddpm_channels_conv=channels,
        ablation_study_F_type='CNN', ablation_study_case='none',
        our_ddpm_clip=100., noise_dist='gaussian', use_xt_residual=False,
        use_volatility_loss=False, use_leverage_loss=False, use_fat_tail_loss=False,
        lambda_energy_max=0., lambda_cross_max=0.,
    )


def decoder(history, future=None):
    if future is None:
        future = torch.zeros(history.shape[0], PRED_LEN, 3, device=history.device)
    return torch.cat([history[:, -21:], future], dim=1)


def batches_forever(loader):
    while True:
        yield from loader


@torch.no_grad()
def validation_score(model, loader, feature_scaler, condition_scaler, max_batches):
    """Full generation, never teacher forcing. Real labels specify the scenario."""
    model.eval()
    scores = []
    # Keep validation noise fixed without perturbing the training RNG stream.
    devices = [model.device.index or 0] if model.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1729)
        for i, (history, _, conditions) in enumerate(loader):
            if i >= max_batches:
                break
            history, conditions = history.to(model.device), conditions.to(model.device)
            supplied = conditions if model.args.market_condition_dim else None
            samples = model(history, None, decoder(history), None,
                            sample_times=1, market_condition=supplied)[0][:, 0]
            raw = feature_scaler.inverse(samples.cpu().numpy())
            actual = condition_scaler.transform(market_statistics(raw[:, :, 0]))
            scores.extend(np.square(actual - conditions.cpu().numpy()).mean(axis=1).tolist())
    return float(np.mean(scores))


def train(cli):
    if cli.ar_steps < 1 or cli.steps < 1 or cli.batch_size < 1 or cli.eval_every < 1 or cli.val_batches < 1:
        raise ValueError('Training steps, batch size and validation settings must be positive')
    out = Path(cli.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a new experiment directory')
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    _, features = load_features(cli.data)
    splits = split_windows(features, cli.train_end, cli.val_end)
    fs = Scaler.fit(features.loc[:cli.train_end, FEATURES].to_numpy())
    raw_conditions = np.array([market_statistics(features.iloc[s:s+PRED_LEN].rclose.to_numpy())
                               for s in splits['train']])
    cs = Scaler.fit(raw_conditions)
    train_data = SPXWindows(features, splits['train'], fs, cs)
    val_data = SPXWindows(features, splits['validation'], fs, cs)
    train_loader = DataLoader(train_data, batch_size=cli.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cli.batch_size, shuffle=False)
    args = model_args(cli.device, cli.channels, not cli.no_market_condition, cli.parameterization, cli.sampler)
    model = Model(args).to(args.device)
    stream = batches_forever(train_loader)
    optimizer = torch.optim.Adam(model.dlinear_model.parameters(), lr=cli.lr)
    for step in range(cli.ar_steps):
        h, y, _ = (v.to(args.device) for v in next(stream))
        optimizer.zero_grad()
        loss = model.pretrain_forward(h, None, decoder(h, y), None)
        loss.backward()
        optimizer.step()
    model.freeze_ar()
    ar_snapshot = {k: v.detach().clone() for k, v in model.dlinear_model.state_dict().items()}
    print(f'AR pretrained for {cli.ar_steps} steps and frozen', flush=True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cli.lr)
    model.diffusion_worker.total_steps = cli.steps
    records, best_score, best_state, best_step = [], float('inf'), None, None
    for step in range(1, cli.steps + 1):
        model.train()
        h, y, c = (v.to(args.device) for v in next(stream))
        optimizer.zero_grad()
        loss = model.train_forward(h, None, decoder(h, y), None,
                                   market_condition=c if args.market_condition_dim else None)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite training loss at step {step}')
        loss.backward()
        optimizer.step()
        if step == 1 or step % cli.eval_every == 0 or step == cli.steps:
            score = validation_score(model, val_loader, fs, cs, cli.val_batches)
            records.append({'step': step, 'train_loss': loss.item(), 'validation_condition_mse': score})
            print(records[-1], flush=True)
            if np.isfinite(score) and score < best_score:
                best_score, best_step = score, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError('No finite validation score; no checkpoint saved')
    if any(not torch.equal(v, ar_snapshot[k]) for k, v in model.dlinear_model.state_dict().items()):
        raise RuntimeError('AR weights changed after freezing')
    saved_args = vars(args).copy()
    saved_args['device'] = str(args.device)
    metadata = {
        'format_version': 1, 'ticker': '^GSPC', 'features': FEATURES,
        'condition_order': ['cumulative_return_21d', 'annualized_log_return_volatility'],
        'volatility_ddof': 1, 'annualization': 252, 'args': saved_args,
        'feature_scaler': fs.state(), 'condition_scaler': cs.state(),
        'train_end': cli.train_end, 'val_end': cli.val_end,
        'split_windows': {k: len(v) for k, v in splits.items()},
        'data_sha256': hashlib.sha256(Path(cli.data).read_bytes()).hexdigest(),
        'seed': cli.seed, 'ar_steps': cli.ar_steps, 'ar_frozen': True,
        'best_step': best_step, 'best_validation_condition_mse': best_score,
        'validation_note': 'Known-future scenario labels; this is not a forecasting score. Overlapping windows are not independent.',
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'metadata': metadata}, out / 'model.pt')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    pd.DataFrame(records).to_csv(out / 'training.csv', index=False)
    pd.DataFrame(raw_conditions, columns=['return', 'volatility']).to_csv(out / 'train_conditions.csv', index=False)
    print(f'Saved {out / "model.pt"}; AR unchanged; test windows unused', flush=True)


def sample(cli):
    if cli.n_paths < 1 or cli.batch_size < 1 or cli.return_tolerance <= 0 or cli.vol_tolerance <= 0:
        raise ValueError('Sample counts and tolerances must be positive')
    targets = np.asarray(cli.condition, dtype=float)
    if not np.isfinite(targets).all() or (targets[:, 0] <= -1).any() or (targets[:, 1] < 0).any():
        raise ValueError('Conditions need finite return > -1 and volatility >= 0')
    out = Path(cli.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a new output directory')
    checkpoint = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    args = SimpleNamespace(**meta['args'])
    args.device = torch.device(cli.device)
    model = Model(args).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.freeze_ar()
    model.eval()
    fs, cs = Scaler(**meta['feature_scaler']), Scaler(**meta['condition_scaler'])
    prices, features = load_features(cli.data)
    history = features.loc[:cli.history_end].iloc[-SEQ_LEN:]
    if len(history) != SEQ_LEN or history.index[-1] != pd.Timestamp(cli.history_end):
        raise ValueError('history_end must be an available trading date with at least 63 feature rows')
    seed_close = float(prices.loc[history.index[-1], 'Close'])
    # No future prices are passed into Model.forward.
    raw_support = pd.read_csv(Path(cli.checkpoint).parent / 'train_conditions.csv').to_numpy()
    support = cs.transform(raw_support)
    out.mkdir(parents=True, exist_ok=True)
    history.reset_index().to_csv(out / 'history_features.csv', index=False)
    summaries, metrics, path_frames = [], [], []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for scenario, target in enumerate(targets):
        torch.manual_seed(cli.seed)  # Same noise across conditions, to isolate condition effects.
        generated = []
        with torch.no_grad():
            for start in range(0, cli.n_paths, cli.batch_size):
                n = min(cli.batch_size, cli.n_paths-start)
                h = torch.tensor(fs.transform(history.to_numpy()), device=args.device).unsqueeze(0).repeat(n, 1, 1)
                c = torch.tensor(cs.transform(target), device=args.device).unsqueeze(0).repeat(n, 1)
                outputs = model(h, None, decoder(h), None, sample_times=1,
                                market_condition=c if args.market_condition_dim else None)[0][:, 0]
                generated.append(fs.inverse(outputs.cpu().numpy()))
        raw = np.concatenate(generated)
        actual = market_statistics(raw[:, :, 0])
        if not np.isfinite(raw).all() or not np.isfinite(actual).all():
            raise RuntimeError('Non-finite generated samples; inspect training before interpreting metrics')
        close = seed_close * np.exp(np.cumsum(raw[:, :, 0], axis=1))
        if not np.isfinite(close).all():
            raise RuntimeError('Generated prices overflowed')
        high_r, low_r = np.maximum(raw[:, :, 1], 0), np.minimum(raw[:, :, 2], 0)
        paths = pd.DataFrame({
            'scenario': scenario, 'path': np.repeat(np.arange(cli.n_paths), PRED_LEN),
            'step': np.tile(np.arange(1, PRED_LEN+1), cli.n_paths),
            'target_return': target[0], 'target_volatility': target[1],
            'rclose': raw[:, :, 0].ravel(), 'rhigh_raw': raw[:, :, 1].ravel(),
            'rlow_raw': raw[:, :, 2].ravel(), 'rhigh': high_r.ravel(), 'rlow': low_r.ravel(),
            'close': close.ravel(), 'high': (close*np.exp(high_r)).ravel(), 'low': (close*np.exp(low_r)).ravel(),
        })
        if not np.isfinite(paths[['close', 'high', 'low']]).all().all():
            raise RuntimeError('Generated high/low prices overflowed')
        path_frames.append(paths)
        errors = np.abs(actual-target)
        hit = (errors[:, 0] <= cli.return_tolerance) & (errors[:, 1] <= cli.vol_tolerance)
        metric = pd.DataFrame(actual, columns=['return', 'volatility'])
        metric.insert(0, 'path', np.arange(cli.n_paths))
        metric.insert(0, 'scenario', scenario)
        metric['hit'] = hit
        metrics.append(metric)
        summaries.append({
            'scenario': scenario, 'target_return': target[0], 'target_volatility': target[1],
            'return_mean': actual[:, 0].mean(), 'volatility_mean': actual[:, 1].mean(),
            'return_mae': errors[:, 0].mean(), 'volatility_mae': errors[:, 1].mean(),
            'return_p05': np.quantile(actual[:, 0], .05), 'return_p95': np.quantile(actual[:, 0], .95),
            'volatility_p05': np.quantile(actual[:, 1], .05), 'volatility_p95': np.quantile(actual[:, 1], .95),
            'joint_hit_rate': hit.mean(), 'rhigh_clip_rate': (raw[:, :, 1] < 0).mean(),
            'rlow_clip_rate': (raw[:, :, 2] > 0).mean(),
            'nearest_train_condition_distance_z': np.linalg.norm(support-cs.transform(target), axis=1).min(),
        })
        for j, ax in enumerate(axes):
            color = f'C{scenario % 10}'
            ax.hist(actual[:, j], bins=min(30, max(3, cli.n_paths//5)), alpha=.35,
                    color=color, label=f'Scenario {scenario}: target {target[j]:.1%}')
            ax.axvline(target[j], color=color, linestyle='--', linewidth=1)
    for ax, title in zip(axes, ['21-day cumulative return', 'Annualized volatility']):
        ax.set_title(title)
        ax.set_ylabel('Number of paths')
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'condition_distributions.png', dpi=150)
    plt.close(fig)
    pd.concat(path_frames, ignore_index=True).to_csv(out / 'paths.csv', index=False)
    pd.concat(metrics, ignore_index=True).to_csv(out / 'path_metrics.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    (out / 'sampling.json').write_text(json.dumps({
        'history_end': cli.history_end, 'seed': cli.seed, 'n_paths': cli.n_paths,
        'return_tolerance': cli.return_tolerance, 'volatility_tolerance': cli.vol_tolerance,
        'checkpoint': str(Path(cli.checkpoint).resolve()), 'conditions': targets.tolist(),
        'market_condition_enabled': bool(args.market_condition_dim),
        'note': 'Finite-sample scenario diagnostics; not RL or forecasting performance.',
    }, indent=2) + '\n')
    print(pd.DataFrame(summaries).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=2)
    sub = parser.add_subparsers(dest='command', required=True)
    data_default = ROOT / 'datasets/spx/prices.csv'
    d = sub.add_parser('download')
    d.add_argument('--start', default='2000-01-01')
    d.add_argument('--end', default='2025-01-01', help='Exclusive Yahoo end date')
    d.add_argument('--data', type=Path, default=data_default)
    t = sub.add_parser('train')
    t.add_argument('--data', type=Path, default=data_default)
    t.add_argument('--out', type=Path, default=ROOT / 'outputs/spx/train')
    t.add_argument('--train-end', required=True)
    t.add_argument('--val-end', required=True)
    t.add_argument('--ar-steps', type=int, default=500)
    t.add_argument('--steps', type=int, default=2000)
    t.add_argument('--eval-every', type=int, default=100)
    t.add_argument('--val-batches', type=int, default=4)
    t.add_argument('--batch-size', type=int, default=32)
    t.add_argument('--channels', type=int, default=64)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--parameterization', choices=['v', 'noise', 'x_start'], default='v')
    t.add_argument('--sampler', choices=['dpm', 'none'], default='dpm')
    t.add_argument('--no-market-condition', action='store_true')
    s = sub.add_parser('sample')
    s.add_argument('--data', type=Path, default=data_default)
    s.add_argument('--checkpoint', type=Path, required=True)
    s.add_argument('--out', type=Path, default=ROOT / 'outputs/spx/samples')
    s.add_argument('--history-end', required=True)
    s.add_argument('--condition', type=float, nargs=2, action='append', required=True, metavar=('RETURN', 'VOL'))
    s.add_argument('--n-paths', type=int, default=200)
    s.add_argument('--batch-size', type=int, default=32)
    s.add_argument('--return-tolerance', type=float, default=.02, help='Absolute return units: .02 = 2 percentage points')
    s.add_argument('--vol-tolerance', type=float, default=.05)
    for p in (t, s):
        p.add_argument('--seed', type=int, default=2024)
        p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    cli = parser.parse_args()
    torch.set_num_threads(cli.threads)
    if cli.command == 'download':
        prices = download_spx(cli.data, cli.start, cli.end)
        print(f'Saved {len(prices)} SPX prices to {cli.data}')
    elif cli.command == 'train':
        train(cli)
    else:
        sample(cli)


if __name__ == '__main__':
    main()
