"""Independently audit this saved experiment without importing its sampler/model."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parent
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'output_layout.py').is_file())
sys.path.insert(0, str(ROOT))
from output_layout import PR30_BATCH, resolve_output_path, verify_historical_hash
BATCH = PR30_BATCH


def digest(path):
    return hashlib.sha256(resolve_output_path(path).read_bytes()).hexdigest()


manifest = json.loads((BATCH / "generation.json").read_text())
run = json.loads((EXPERIMENT / "run_metadata.json").read_text())
assert manifest["status"] == "complete" and manifest["ppo_run"] is False
for name, expected in manifest["artifacts_sha256"].items():
    verify_historical_hash(BATCH / name, expected)
for name, expected in run["existing_inputs_sha256"].items():
    assert digest(ROOT / name) == expected, name
assert digest(manifest["checkpoint"]) == manifest["checkpoint_sha256"]
assert digest(manifest["data"]) == manifest["data_sha256"]
z = np.load(BATCH / "paths.npz", allow_pickle=False)
plan = pd.read_csv(BATCH / "sampling_plan.csv")
population = pd.read_csv(BATCH / "population.csv")
metrics = pd.read_csv(BATCH / "path_metrics.csv")
diagnostics = json.loads((BATCH / "diagnostics.json").read_text())
prices = pd.read_csv(manifest["data"], parse_dates=["date"])
tickers = z["tickers"].tolist()
w = {k: prices.pivot(index="date", columns="tic", values=k)[tickers] for k in ("close", "high", "low")}
rc = np.log(w["close"] / w["close"].shift(1)).to_numpy()
rh = np.log(w["high"] / w["close"]).to_numpy()
rl = np.log(w["low"] / w["close"]).to_numpy()
features = np.stack([rc, rh, rl], axis=2).reshape(len(rc), -1)[1:]
dates = w["close"].index[1:]
ids = z["draw_id"]
assert len(plan) == 500 and len(ids) == 2000
assert (plan.source_pool == "full").sum() == 350
assert (plan.source_pool == "tail").sum() == 150
assert (pd.to_datetime(plan.target_end) <= "2017-12-31").all()
assert (pd.to_datetime(plan.target_start) >= "2010-01-01").all()
assert plan.groupby("draw_id").size().eq(1).all()
assert np.array_equal(ids, np.repeat(plan.draw_id.to_numpy(), 4))
assert np.array_equal(z["replicate_id"], np.tile(np.arange(4), len(plan)))
assert np.array_equal(z["anchor_dates"], plan.anchor_date.to_numpy()[ids])
starts = dates.get_indexer(pd.to_datetime(plan.anchor_date)) + 1
np.testing.assert_array_equal(starts, plan.window_id)
histories = np.stack([features[s-63:s] for s in starts])
futures = np.stack([features[s:s+21] for s in starts])
np.testing.assert_allclose(z["history_features"], histories, atol=1e-8, rtol=1e-6)
np.testing.assert_allclose(z["paired_real_future"], futures, atol=1e-8, rtol=1e-6)
gross = np.exp(futures[..., ::3]).mean(axis=2)
targets = np.column_stack([np.prod(gross, axis=1)-1, np.log(gross).std(axis=1, ddof=1)*np.sqrt(252)])
np.testing.assert_allclose(z["target"], targets[ids], atol=1e-12, rtol=1e-12)
anchors = w["close"].loc[pd.to_datetime(z["anchor_dates"])].to_numpy()
np.testing.assert_array_equal(z["anchor_close"], anchors)
close = z["close"].astype(np.float64)
prices_with_anchor = np.concatenate([anchors[:, None, :], close], axis=1)
gross_assets = prices_with_anchor[:, 1:] / prices_with_anchor[:, :-1]
np.testing.assert_allclose(np.log(gross_assets), z["raw_features"][..., ::3], atol=2e-7, rtol=1e-4)
gross_portfolio = gross_assets.mean(axis=2)
actual = np.column_stack([np.prod(gross_portfolio, axis=1)-1,
                          np.log(gross_portfolio).std(axis=1, ddof=1)*np.sqrt(252)])
max_errors = np.max(np.abs(actual-z["realized"]), axis=0)
np.testing.assert_allclose(actual, z["realized"], atol=2e-6, rtol=1e-5)
np.testing.assert_allclose(z["realized"], metrics[["return_21d", "volatility_ann"]], atol=1e-12)
assert (z["low"] <= z["close"]).all() and (z["close"] <= z["high"]).all()
assert all(np.isfinite(z[k]).all() and (z[k] > 0).all() for k in ("close", "high", "low"))
unique = len(np.unique(z["raw_features"][..., ::3].reshape(len(ids), -1), axis=0))
assert unique == diagnostics["diversity"]["n_unique_close_paths"]
cutoff = np.quantile(population.target_return, .3)
assert float((plan.target_return <= cutoff).mean()) == diagnostics["target_tail_share"]
assert float((z["realized"][:, 0] <= cutoff).mean()) == diagnostics["realized_tail_share"]
result = {
    "status": "passed", "n_paths": len(ids), "n_draws": len(plan),
    "n_artifact_hashes_verified": len(manifest["artifacts_sha256"]),
    "n_original_input_hashes_unchanged": len(run["existing_inputs_sha256"]),
    "history_and_future_pairing_verified": True,
    "n_conditions_independently_recomputed": len(plan),
    "max_return_error_from_saved_prices": float(max_errors[0]),
    "max_volatility_error_from_saved_prices": float(max_errors[1]),
    "n_unique_close_paths": unique,
    "target_tail_share": diagnostics["target_tail_share"],
    "realized_tail_share": diagnostics["realized_tail_share"],
    "scope": "Saved data integrity, pairing and independent numerical consistency; no PPO/out-of-sample claim.",
}
with (EXPERIMENT / "batch_audit.json").open("x") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
