"""Compare frozen PR30 and new PR10 batches on identical training thresholds.

Run from TimeDiff: .venv-spx/bin/python outputs/dow10/conditional/empirical_joint/run_20261017/analyses/pr30_vs_pr10/analyze.py
This script performs no model inference, training, filtering, or PPO.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'output_layout.py').is_file())
sys.path.insert(0, str(ROOT))
from output_layout import PR10_BATCH, resolve_output_path, verify_historical_hash
from data_provider.dow10_cond import COLUMNS, CLOSE_INDICES, load_features
from evaluation.joint_condition_diagnostics import path_statistics

BATCHES = {"PR30": ROOT / "outputs/dow10/conditional/empirical_joint/run_20261017/batches/full70_tail30_pr30", "PR10": PR10_BATCH}


def sha(path):
    h = hashlib.sha256()
    with resolve_output_path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def structure(raw):
    raw = np.asarray(raw, dtype=np.float64)
    metrics, _ = path_statistics(raw)
    close = raw[..., CLOSE_INDICES]
    centered = close - close.mean(axis=1, keepdims=True)
    cov = np.einsum("nti,ntj->nij", centered, centered) / (close.shape[1] - 1)
    var = np.diagonal(cov, axis1=1, axis2=2)
    corr = cov / np.sqrt(var[:, :, None] * var[:, None, :])
    metrics["asset_volatility"] = np.sqrt(var).mean(axis=1) * np.sqrt(252)
    metrics["common_variance"] = close.mean(axis=2).var(axis=1, ddof=1)
    metrics["residual_variance"] = (close - close.mean(axis=2, keepdims=True)).var(axis=1, ddof=1).mean(axis=1)
    np.testing.assert_allclose(var.mean(axis=1), metrics.common_variance + metrics.residual_variance)
    metrics["high_invalid"] = (raw[..., 1::3] < 0).mean(axis=(1, 2))
    metrics["low_invalid"] = (raw[..., 2::3] > 0).mean(axis=(1, 2))
    return metrics, corr


def analyze():
    manifests = {name: json.loads((p / "generation.json").read_text()) for name, p in BATCHES.items()}
    assert all(m["status"] == "complete" and not m["ppo_run"] for m in manifests.values())
    assert len({m["checkpoint_sha256"] for m in manifests.values()}) == 1
    assert len({m["data_sha256"] for m in manifests.values()}) == 1
    protected = {}
    for name, path in BATCHES.items():
        m = manifests[name]
        for f, expected in m["artifacts_sha256"].items():
            verify_historical_hash(path / f, expected)
        for f, expected in m["source_sha256"].items():
            verify_historical_hash(ROOT / f, expected)
        for f, expected in [(m["data"], m["data_sha256"]), (m["checkpoint"], m["checkpoint_sha256"])]:
            assert sha(f) == expected, f
        protected[name] = {"manifest_sha256": sha(path / "generation.json"), "paths_sha256": sha(path / "paths.npz")}
    pop = pd.read_csv(BATCHES["PR30"] / "population.csv")
    newpop = pd.read_csv(BATCHES["PR10"] / "population.csv")
    pd.testing.assert_frame_equal(pop.drop(columns=["is_tail", "mixture_probability"]), newpop.drop(columns=["is_tail", "mixture_probability"]))
    _, features = load_features(manifests["PR30"]["data"])
    feature_values = features[COLUMNS].to_numpy()
    population_raw = np.stack([feature_values[s:s+21] for s in pop.window_id])
    popstats, _ = structure(population_raw)
    thresholds = {f"PR{int(q*100)}": float(np.quantile(pop.target_return, q)) for q in (.01, .05, .10, .30)}
    assert abs(thresholds["PR10"] - manifests["PR10"]["sampling"]["tail_return_cutoff"]) < 1e-12
    tails, summaries, cohorts, pairs, coverage, reconstruction = [], [], [], [], [], []
    loaded = {}
    for name, path in BATCHES.items():
        z = np.load(path / "paths.npz", allow_pickle=False)
        plan = pd.read_csv(path / "sampling_plan.csv")
        diag = json.loads((path / "diagnostics.json").read_text())
        targets = z["target"]
        raw = z["raw_features"].astype(np.float64)
        ids = z["draw_id"]
        original_refs = np.stack([feature_values[s:s+21] for s in plan.window_id])
        original_histories = np.stack([feature_values[s-63:s] for s in plan.window_id])
        np.testing.assert_array_equal(original_refs.astype(np.float32), z["paired_real_future"])
        np.testing.assert_array_equal(original_histories.astype(np.float32), z["history_features"])
        gen, gen_corr = structure(raw)
        real, real_corr = structure(original_refs[ids])
        np.testing.assert_allclose(real[["return_21d", "volatility_ann"]], targets, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(gen[["return_21d", "volatility_ann"]], z["realized"], atol=1e-12)
        assert np.isfinite(raw).all() and len(raw) == 2000
        assert (z["low"] <= z["close"]).all() and (z["close"] <= z["high"]).all()
        assert all(np.isfinite(z[k]).all() and (z[k] > 0).all() for k in ("close", "high", "low"))
        assert plan.target_end.max() <= "2017-12-31"
        error = gen[["return_21d", "volatility_ann"]].to_numpy() - targets
        for cutoff_name, cutoff in thresholds.items():
            target_tail = targets[:, 0] <= cutoff
            generated_tail = gen.return_21d.to_numpy() <= cutoff
            kept = int((target_tail & generated_tail).sum())
            lost = int((target_tail & ~generated_tail).sum())
            gained = int((~target_tail & generated_tail).sum())
            pool = pop if name == "PR30" else newpop
            row = {"batch": name, "threshold": cutoff_name, "cutoff": cutoff,
                   "expected_share": float(np.dot(pool.target_return <= cutoff, pool.mixture_probability)),
                   "target_count": int(target_tail.sum()), "generated_count": int(generated_tail.sum()),
                   "target_share": float(target_tail.mean()), "generated_share": float(generated_tail.mean()),
                   "kept": kept, "lost": lost, "gained": gained,
                   "retention": kept / int(target_tail.sum()) if target_tail.any() else None}
            assert row["target_count"] - lost + gained == row["generated_count"]
            tails.append(row)
        for source in ("all", "full", "tail"):
            mask = np.ones(len(raw), dtype=bool) if source == "all" else z["source_pool"] == source
            g, r = gen.loc[mask], real.loc[mask]
            row = {"batch": name, "source": source, "n_paths": int(mask.sum()),
                   "generated_return_mean": float(g.return_21d.mean()), "target_return_mean": float(targets[mask, 0].mean()),
                   "generated_vol_mean": float(g.volatility_ann.mean()), "target_vol_mean": float(targets[mask, 1].mean()),
                   "return_mae": float(np.abs(error[mask, 0]).mean()), "return_bias": float(error[mask, 0].mean()),
                   "volatility_mae": float(np.abs(error[mask, 1]).mean()),
                   "generated_corr": float(g.mean_asset_correlation.mean()), "paired_real_corr": float(r.mean_asset_correlation.mean()),
                   "corr_gap": float(g.mean_asset_correlation.mean() - r.mean_asset_correlation.mean()),
                   "high_invalid": float(g.high_invalid.mean()), "low_invalid": float(g.low_invalid.mean()),
                   "generated_drawdown_mean": float(g.max_drawdown.mean()), "paired_real_drawdown_mean": float(r.max_drawdown.mean()),
                   "asset_vol_ratio": float(g.asset_volatility.mean() / r.asset_volatility.mean()),
                   "residual_variance_ratio": float(g.residual_variance.mean() / r.residual_variance.mean())}
            cohorts.append(row)
            if source == "all":
                summaries.append({**row, **{f"diversity_{k}": v for k, v in diag["diversity"].items() if isinstance(v, (int, float))},
                                  "n_unique_windows": int(plan.window_id.nunique()),
                                  "expected_mixture_corr": float(np.dot(popstats.mean_asset_correlation, (pop if name == "PR30" else newpop).mixture_probability))})
        rc, gc = real_corr.mean(axis=0), gen_corr.mean(axis=0)
        for i in range(10):
            for j in range(i+1, 10):
                pairs.append({"batch": name, "asset1": str(z["tickers"][i]), "asset2": str(z["tickers"][j]),
                              "paired_real": float(rc[i,j]), "generated": float(gc[i,j]), "gap": float(gc[i,j]-rc[i,j])})
        for source in ("all", "full", "tail"):
            sub = plan if source == "all" else plan[plan.source_pool == source]
            counts = sub.window_id.value_counts()
            coverage.append({"batch": name, "source": source, "n_draws": len(sub), "n_unique_windows": len(counts),
                             "max_repeats_same_window": int(counts.max()),
                             "largest_year_share": float(sub.target_start.str[:4].value_counts(normalize=True).max()),
                             "largest_year": str(sub.target_start.str[:4].value_counts().index[0])})
        close = z["close"].astype(float)
        with_anchor = np.concatenate([z["anchor_close"][:, None], close], axis=1)
        rebuilt = raw.copy()
        rebuilt[..., CLOSE_INDICES] = np.log(with_anchor[:, 1:] / with_anchor[:, :-1])
        rebuilt_stats, rebuilt_corr = structure(rebuilt)
        reconstruction.append({"batch": name,
            "max_return_difference": float(np.abs(rebuilt_stats.return_21d-gen.return_21d).max()),
            "max_corr_difference": float(np.abs(rebuilt_corr-gen_corr).max()),
            **{f"{key}_membership_changes": int(((rebuilt_stats.return_21d <= cutoff) != (gen.return_21d <= cutoff)).sum()) for key, cutoff in thresholds.items()}})
        loaded[name] = {"z": z, "plan": plan, "gen": gen, "targets": targets}
    old, new = loaded["PR30"], loaded["PR10"]
    assert np.array_equal(old["z"]["source_pool"], new["z"]["source_pool"])
    full = old["z"]["source_pool"] == "full"
    full_targets_equal = np.array_equal(old["targets"][full], new["targets"][full])
    full_raw_equal = np.array_equal(old["z"]["raw_features"][full], new["z"]["raw_features"][full])
    assert full_targets_equal and full_raw_equal
    tables = {"tail_comparison": tails, "summary_comparison": summaries, "cohort_comparison": cohorts,
              "pairwise_correlation": pairs, "coverage_comparison": coverage, "reconstruction_audit": reconstruction}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    result = {"status": "complete", "ppo_run": False, "new_model_training": False,
              "thresholds": thresholds, "n_paths_per_batch": 2000,
              "comparison": "70% full + 30% portfolio PR30 versus 70% full + 30% portfolio PR10",
              "full_component_control": {"n_paths": int(full.sum()), "targets_identical": full_targets_equal,
                                          "raw_features_bitwise_identical": full_raw_equal},
              "tail_comparison": tails, "summary_comparison": summaries,
              "coverage_comparison": coverage, "reconstruction_audit": reconstruction,
              "source_hashes": protected,
              "limitations": ["One checkpoint, one sampling seed and one diffusion seed; no out-of-sample or PPO evidence.",
                              "Overlapping windows and repeated conditions are dependent; 2000 paths are not 2000 independent events.",
                              "This is a sampler change, not a model training improvement.",
                              "No generated paths were filtered, relabeled or retuned."]}
    for name, path in BATCHES.items():
        assert sha(path / "generation.json") == protected[name]["manifest_sha256"]
        assert sha(path / "paths.npz") == protected[name]["paths_sha256"]
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"thresholds": thresholds, "tail_comparison": tails, "summary_comparison": summaries,
                      "full_component_control": result["full_component_control"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    analyze()
