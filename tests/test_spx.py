import numpy as np
import pandas as pd
import pytest
import torch

from data_provider.spx import (Scaler, SPXWindows, load_features, market_statistics,
                               split_windows)
from models_diffusion.DDPM import Model
from spx_experiment import decoder, main, model_args

torch.set_num_threads(1)


def test_statistics_include_first_target_return():
    returns = np.log(np.array([1.01] + [1.] * 20))
    actual = market_statistics(returns)
    assert actual[0] == pytest.approx(.01)
    assert actual[1] == pytest.approx(np.std(returns, ddof=1) * np.sqrt(252))


def prices_fixture(tmp_path):
    rng = np.random.default_rng(12)
    dates = pd.bdate_range('2019-01-01', periods=360)
    close = 100 * np.exp(np.cumsum(rng.normal(.0003, .01, len(dates))))
    frame = pd.DataFrame({'date': dates, 'Close': close, 'High': close*1.01, 'Low': close*.99})
    path = tmp_path / 'prices.csv'
    frame.to_csv(path, index=False)
    return path


def test_split_labels_and_scalers(tmp_path):
    _, features = load_features(prices_fixture(tmp_path))
    train_end, val_end = features.index[179], features.index[269]
    split = split_windows(features, train_end, val_end)
    for s in split['train']:
        assert features.index[s+20] <= train_end
    for s in split['validation']:
        assert train_end < features.index[s] <= features.index[s+20] <= val_end
    for s in split['test']:
        assert features.index[s] > val_end
    fs = Scaler.fit(features.loc[:train_end].to_numpy())
    train_labels = np.array([market_statistics(features.iloc[s:s+21].rclose) for s in split['train']])
    cs = Scaler.fit(train_labels)
    ds = SPXWindows(features, split['train'], fs, cs)
    h, y, c = ds[0]
    assert h.shape == (63, 3) and y.shape == (21, 3)
    np.testing.assert_allclose(cs.inverse(c), market_statistics(fs.inverse(y)[:, 0]), atol=1e-7)
    changed = features.copy()
    changed.loc[changed.index > train_end, :] = 999
    other = Scaler.fit(changed.loc[:train_end].to_numpy())
    np.testing.assert_array_equal(fs.mean, other.mean)


@pytest.mark.parametrize('sampler', ['dpm', 'none'])
@pytest.mark.parametrize('parameterization', ['x_start', 'noise', 'v'])
def test_frozen_ar_condition_gradient_and_sampling(sampler, parameterization):
    torch.manual_seed(123)
    args = model_args(channels=8, sampler=sampler, parameterization=parameterization)
    args.diff_steps, args.sampling_steps = 12, 4
    model = Model(args)
    history, target, condition = torch.randn(3, 63, 3), torch.randn(3, 21, 3), torch.randn(3, 2)
    opt_ar = torch.optim.Adam(model.dlinear_model.parameters())
    model.pretrain_forward(history, None, decoder(history, target), None).backward()
    opt_ar.step()
    opt_ar.zero_grad(set_to_none=True)
    model.freeze_ar()
    saved = {k: v.clone() for k, v in model.dlinear_model.state_dict().items()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    loss = model.train_forward(history, None, decoder(history, target), None, market_condition=condition)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.diffusion_worker.nn.market_film[0].weight.grad.abs().sum() > 0
    opt.step()
    assert all(torch.equal(saved[k], v) for k, v in model.dlinear_model.state_dict().items())
    assert all(p.grad is None for p in model.dlinear_model.parameters())
    model.eval()
    with torch.no_grad():
        torch.manual_seed(21)
        output = model(history, None, decoder(history), None, sample_times=1, market_condition=condition)[0]
        torch.manual_seed(21)
        changed = model(history, None, decoder(history), None, sample_times=1, market_condition=condition+2)[0]
        # Real future arguments must not affect generated samples (no inference mixup).
        torch.manual_seed(21)
        other_future = model(history, None, decoder(history, target*100), None, sample_times=1, market_condition=condition)[0]
    assert output.shape == (3, 1, 21, 3) and torch.isfinite(output).all()
    assert not torch.allclose(output, changed)
    torch.testing.assert_close(output, other_future)


def test_legacy_unconditioned_api_and_required_condition():
    h = torch.randn(1, 63, 3)
    args = model_args(channels=8, conditioned=False)
    model = Model(args).eval()
    with torch.no_grad():
        result = model(h, None, decoder(h), None, sample_times=1)[0]
    assert result.shape == (1, 1, 21, 3)
    with pytest.raises(ValueError, match='without market'):
        model(h, None, decoder(h), None, sample_times=1, market_condition=torch.zeros(1, 2))
    conditioned = Model(model_args(channels=8))
    with pytest.raises(ValueError, match='market_condition'):
        conditioned(h, None, decoder(h), None, sample_times=1)


@pytest.mark.parametrize('future_mixup', [False, True, None])
def test_future_conditioning_ablation(future_mixup):
    """At fixed x_t, labels affect conditioning only in the explicit/legacy baseline."""
    torch.manual_seed(7)
    args = model_args(channels=8, future_mixup=future_mixup)
    if future_mixup is None:
        del args.use_future_mixup  # Older checkpoints retain original behavior.
    net = Model(args).diffusion_worker.nn.eval()
    noisy = torch.randn(2, 3, 21)
    condition = {'history': torch.randn(2, 3, 84), 'market': torch.randn(2, 2)}
    timestep = torch.tensor([10, 80])
    clean = torch.randn_like(noisy, requires_grad=True)
    torch.manual_seed(9)
    first, _ = net(noisy, timestep, condition, y_clean=clean)
    torch.manual_seed(9)
    second, _ = net(noisy, timestep, condition, y_clean=clean + 10)
    if future_mixup is False:
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        inference = net(noisy, timestep, condition, y_clean=None)
        torch.testing.assert_close(first, inference, rtol=0, atol=0)
        assert torch.autograd.grad(first.sum(), clean, allow_unused=True)[0] is None
    else:
        assert not torch.allclose(first, second)
        assert torch.autograd.grad(first.sum(), clean)[0].abs().sum() > 0


def test_cli_round_trip(tmp_path, monkeypatch):
    data = prices_fixture(tmp_path)
    _, features = load_features(data)
    checkpoint_dir, output_dir = tmp_path/'train', tmp_path/'samples'
    monkeypatch.setattr('sys.argv', [
        'spx_experiment.py', 'train', '--data', str(data), '--out', str(checkpoint_dir),
        '--train-end', str(features.index[179].date()), '--val-end', str(features.index[269].date()),
        '--steps', '2', '--ar-steps', '2', '--eval-every', '2', '--val-batches', '1',
        '--channels', '8', '--batch-size', '3', '--device', 'cpu', '--no-progress',
    ])
    main()
    ckpt = torch.load(checkpoint_dir/'model.pt', weights_only=True)
    assert ckpt['metadata']['ar_frozen']
    assert ckpt['metadata']['args']['use_future_mixup'] is False
    assert ckpt['metadata']['training_config']['steps'] == 2
    monkeypatch.setattr('sys.argv', [
        'spx_experiment.py', 'sample', '--data', str(data), '--out', str(output_dir),
        '--checkpoint', str(checkpoint_dir/'model.pt'), '--history-end', str(features.index[269].date()),
        '--condition', '-0.05', '0.3', '--condition', '0.05', '0.3',
        '--n-paths', '5', '--batch-size', '3', '--device', 'cpu', '--no-progress',
    ])
    main()
    paths = pd.read_csv(output_dir/'paths.csv')
    assert len(paths) == 2*5*21
    assert (paths.low <= paths.close).all() and (paths.close <= paths.high).all()
    metrics = pd.read_csv(output_dir/'path_metrics.csv')
    actual = market_statistics(paths.query('scenario == 0 and path == 0').rclose.to_numpy())
    np.testing.assert_allclose(metrics.iloc[0][['return', 'volatility']].to_numpy(dtype=float), actual)
    assert (output_dir/'condition_distributions.png').is_file()
