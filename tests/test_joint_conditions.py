import json

import numpy as np
import pandas as pd
import pytest
import torch

from data_provider.dow10_cond import (
    CLOSE_INDICES, COLUMNS, TICKERS, Scaler, load_features, portfolio_statistics,
)
from data_provider.joint_conditions import make_joint_plan, paired_inputs, training_population
from dow10_cond_experiment import model_args
from evaluation.joint_condition_diagnostics import diversity_statistics
from models_diffusion.DDPM import Model
from sample_joint_dow10 import generate_raw, load_conditioned_model, main, sha256


def features_fixture():
    rng = np.random.default_rng(11)
    values = rng.normal(.0004, .01, (420, 30))
    return pd.DataFrame(values, index=pd.bdate_range("2016-09-01", periods=420), columns=COLUMNS)


def test_exact_mixture_preserves_history_return_vol_pairing():
    f = features_fixture()
    plan = make_joint_plan(f, n_draws=100, seed=9, train_start="2017-01-01")
    assert plan.draws.source_pool.value_counts().to_dict() == {"full": 70, "tail": 30}
    assert plan.draws.loc[plan.draws.source_pool == "tail", "is_tail"].all()
    assert plan.draws.loc[plan.draws.source_pool == "full", "is_tail"].any()
    assert plan.population.mixture_probability.sum() == pytest.approx(1)
    expected_tail = (plan.population.mixture_probability * plan.population.is_tail).sum()
    assert expected_tail == pytest.approx(plan.metadata["expected_sampled_tail_share"])
    histories, futures, targets = paired_inputs(f, plan.draws)
    for i, row in enumerate(plan.draws.itertuples()):
        np.testing.assert_array_equal(histories[i], f.iloc[row.window_id-63:row.window_id].to_numpy())
        assert row.anchor_date == str(f.index[row.window_id-1].date())
        # Independent direct gross-return formula, not the helper under test.
        rp = np.log(np.exp(futures[i][:, CLOSE_INDICES]).mean(axis=1))
        np.testing.assert_allclose(targets[i], [np.exp(rp.sum())-1, rp.std(ddof=1)*np.sqrt(252)], atol=1e-12)
        assert row.target_start >= "2017-01-01" and row.target_end <= "2017-12-31"


def test_sampling_is_seeded_and_cannot_read_later_periods():
    f = features_fixture()
    kwargs = dict(n_draws=100, seed=13, train_start="2017-01-01")
    first = make_joint_plan(f, **kwargs)
    changed = f.copy()
    changed.loc[changed.index > "2017-12-31", :] = np.nan
    second = make_joint_plan(changed, **kwargs)
    pd.testing.assert_frame_equal(first.draws, second.draws)
    pd.testing.assert_frame_equal(first.population, second.population)
    assert first.metadata == second.metadata
    other = make_joint_plan(f, **{**kwargs, "seed": 14})
    assert not first.draws.window_id.equals(other.draws.window_id)


def test_reject_changed_pair_and_unsorted_dates():
    f = features_fixture()
    plan = make_joint_plan(f, n_draws=10, train_start="2017-01-01")
    f.iloc[int(plan.draws.window_id.iloc[0]), 0] += .1
    with pytest.raises(ValueError, match="no longer match"):
        paired_inputs(f, plan.draws)
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        training_population(f.iloc[::-1], "2017-01-01", "2017-12-31")


def test_cutoff_ties_and_rounded_pool_counts_are_explicit():
    f = features_fixture() * 0
    p = make_joint_plan(f, n_draws=11, train_start="2017-01-01")
    assert p.metadata["n_full_draws"] == 8
    assert p.metadata["n_tail_draws"] == 3
    assert p.metadata["population_tail_share"] == 1
    assert p.metadata["expected_sampled_tail_share"] == pytest.approx(1)


@pytest.mark.parametrize("kwargs", [{"full_weight": 1.1}, {"tail_quantile": 0}, {"n_draws": 0}])
def test_invalid_sampling_controls(kwargs):
    with pytest.raises(ValueError):
        make_joint_plan(features_fixture(), train_start="2017-01-01", **kwargs)


def test_diversity_detects_clones_and_shape_changes_at_same_condition():
    rng = np.random.default_rng(23)
    base = rng.normal(0, .01, (1, 21, 30))
    clones = np.repeat(base, 4, axis=0)
    metadata = pd.DataFrame({"window_id": [63]*4, "anchor_date": ["2017-01-01"]*4,
                             "target_return": [0.]*4, "target_volatility": [.2]*4})
    table, s = diversity_statistics(clones, clones, metadata, np.ones(30))
    assert s["exact_duplicate_close_path_fraction"] == .75
    assert s["n_conditions_all_close_paths_identical"] == 1
    assert table.mean_pairwise_normalized_rmse.iloc[0] == 0
    assert s["original_future_copy_count_at_1e_minus_7"] == 4
    # Permuting whole daily vectors preserves R and sigma but changes the path.
    varied = np.stack([np.roll(base[0], j, axis=0) for j in range(4)])
    np.testing.assert_allclose(portfolio_statistics(varied[..., CLOSE_INDICES]),
                               np.repeat(portfolio_statistics(base[..., CLOSE_INDICES]), 4, axis=0), atol=1e-12)
    _, s = diversity_statistics(varied, clones, metadata, np.ones(30))
    assert s["exact_duplicate_close_path_fraction"] == 0
    assert s["median_pairwise_centered_rmse"] > 0


def test_generation_never_passes_real_future_to_decoder():
    class Spy:
        device = torch.device("cpu")
        def __call__(self, h, _, decoder_input, __, sample_times, market_condition):
            assert torch.count_nonzero(decoder_input[:, -21:]) == 0
            assert market_condition.shape == (len(h), 2)
            return (torch.randn(len(h), 1, 21, 30),)
    histories = np.zeros((3, 63, 30))
    fs, cs = Scaler(np.zeros(30), np.ones(30)), Scaler(np.zeros(2), np.ones(2))
    a, ids = generate_raw(Spy(), histories, np.zeros((3, 2)), fs, cs,
                          paths_per_condition=4, batch_size=5, diffusion_seed=4)
    b, _ = generate_raw(Spy(), histories, np.zeros((3, 2)), fs, cs,
                        paths_per_condition=4, batch_size=5, diffusion_seed=4)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ids, np.repeat(np.arange(3), 4))


def checkpoint_fixture(tmp_path):
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2016-01-01", periods=280)
    close = 100*np.exp(np.cumsum(rng.normal(.0003, .01, (len(dates), 10)), axis=0))
    data = tmp_path / "prices.csv"
    pd.DataFrame({"date": np.repeat(dates, 10), "tic": np.tile(TICKERS, len(dates)),
                  "close": close.ravel(), "high": (close*1.01).ravel(), "low": (close*.99).ravel()}).to_csv(data, index=False)
    _, f = load_features(data)
    start, end = str(f.index[63].date()), str(f.index[200].date())
    population, _ = training_population(f, start, end)
    fs = Scaler.fit(f.loc[:end].to_numpy())
    cs = Scaler.fit(population[["target_return", "target_volatility"]].to_numpy())
    args = model_args("cpu", channels=8)
    args.diff_steps, args.sampling_steps = 8, 3
    torch.manual_seed(4)
    model = Model(args)
    saved_args = {**vars(args), "device": "cpu"}
    meta = {"universe": "dow10", "columns": COLUMNS, "close_indices": list(CLOSE_INDICES),
            "target_start": start, "train_end": end, "val_end": "2017-01-27", "args": saved_args,
            "feature_scaler": fs.state(), "condition_scaler": cs.state(), "data_sha256": sha256(data)}
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": meta}, checkpoint)
    return data, checkpoint, start, end


def test_checkpoint_to_artifacts_and_non_overwrite(tmp_path):
    torch.set_num_threads(1)
    data, checkpoint, start, end = checkpoint_fixture(tmp_path)
    before = (sha256(data), sha256(checkpoint))
    out = tmp_path / "batch"
    argv = ["generate", "--data", str(data), "--checkpoint", str(checkpoint), "--out", str(out),
            "--train-start", start, "--train-end", end, "--n-draws", "10",
            "--paths-per-condition", "2", "--batch-size", "3", "--threads", "1"]
    main(argv)
    m = json.loads((out / "generation.json").read_text())
    assert m["status"] == "complete" and m["ppo_run"] is False
    assert m["sampling"]["n_full_draws"] == 7
    z = np.load(out / "paths.npz", allow_pickle=False)
    assert z["raw_features"].shape == (20, 21, 30)
    assert z["history_features"].shape == (10, 63, 30)
    assert z["close"].shape == (20, 21, 10)
    np.testing.assert_allclose(z["close"][:, 0]/np.exp(z["raw_features"][:, 0, ::3]), z["anchor_close"], rtol=1e-5)
    assert (z["low"] <= z["close"]).all() and (z["close"] <= z["high"]).all()
    np.testing.assert_allclose(z["realized"], portfolio_statistics(z["raw_features"][..., CLOSE_INDICES]))
    assert (out / "condition_distributions.png").stat().st_size > 1000
    assert (out / "within_condition_paths.png").stat().st_size > 1000
    for name, digest in m["artifacts_sha256"].items():
        assert sha256(out / name) == digest
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(argv)
    assert (sha256(data), sha256(checkpoint)) == before


def test_reject_wrong_data_and_later_trained_checkpoint(tmp_path):
    data, checkpoint, start, end = checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_conditioned_model(checkpoint, device="cpu", train_start=start, train_end=end, data_sha256="wrong")
    with pytest.raises(ValueError, match="train_end must match"):
        load_conditioned_model(checkpoint, device="cpu", train_start=start, train_end="2025-01-01", data_sha256=sha256(data))
