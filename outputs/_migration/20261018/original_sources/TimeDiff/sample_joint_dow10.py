"""Generate paired empirical Dow 10 conditions and inspect paths, without PPO.

Example (all outputs must be a new directory):
  python sample_joint_dow10.py generate --checkpoint outputs/new_train/model.pt \
    --data datasets/dow10/prices_2004.csv --out outputs/new_joint_batch
  python sample_joint_dow10.py plan --data datasets/dow10/prices_2004.csv \
    --out outputs/new_joint_plan
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data_provider.dow10_cond import (
    CLOSE_INDICES, COLUMNS, N_VARS, PRED_LEN, SEQ_LEN, TICKERS,
    Scaler, load_features, portfolio_statistics, wide_prices,
)
from data_provider.joint_conditions import make_joint_plan, paired_inputs
from dow10_cond_experiment import decoder, reconstruct_prices

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_metadata(data):
    files = ["sample_joint_dow10.py", "data_provider/joint_conditions.py",
             "data_provider/dow10_cond.py", "evaluation/joint_condition_diagnostics.py",
             "dow10_cond_experiment.py", "models_diffusion/DDPM.py",
             "models_diffusion/DDPM_modules/DDPM_CNNNet.py",
             "models_diffusion/diffusion_worker.py", "models_diffusion/samplers/dpm_sampler.py",
             "models_diffusion/samplers/dpm_solver.py"]
    return {
        "data": str(Path(data).resolve()), "data_sha256": sha256(data),
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip() or None,
        "source_sha256": {name: sha256(ROOT / name) for name in files},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }


def load_conditioned_model(checkpoint, *, device, train_start, train_end, data_sha256):
    import torch
    from models_diffusion.DDPM import Model
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not {"metadata", "state_dict"} <= saved.keys():
        raise ValueError("Need a conditional Dow 10 checkpoint with metadata and state_dict")
    meta = saved["metadata"]
    if meta.get("universe") != "dow10" or meta.get("columns") != COLUMNS:
        raise ValueError("Checkpoint must use the same ordered Dow 10 feature columns")
    if meta.get("close_indices") != list(CLOSE_INDICES):
        raise ValueError("Checkpoint close indices do not match the feature layout")
    if pd.Timestamp(train_start) < pd.Timestamp(meta["target_start"]):
        raise ValueError("Sampling starts before the generator's training target period")
    if pd.Timestamp(train_end) != pd.Timestamp(meta["train_end"]):
        raise ValueError("Sampling train_end must match generator train_end; do not use later-trained models")
    if meta.get("data_sha256") != data_sha256:
        raise ValueError("Input data SHA-256 differs from the checkpoint's training data")
    args = SimpleNamespace(**meta["args"])
    if (args.num_vars, args.seq_len, args.pred_len, getattr(args, "market_condition_dim", 0)) != (N_VARS, SEQ_LEN, PRED_LEN, 2):
        raise ValueError("Expected 30 channels, 63 history days, 21 future days and 2 market conditions")
    if getattr(args, "vis_ar_part", False):
        raise ValueError("Checkpoint visualization side effects must be disabled")
    fs, cs = Scaler(**meta["feature_scaler"]), Scaler(**meta["condition_scaler"])
    for scaler, size in ((fs, N_VARS), (cs, 2)):
        if (scaler.mean.shape != (size,) or scaler.scale.shape != (size,)
                or not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all()
                or (scaler.scale <= 0).any()):
            raise ValueError("Checkpoint has invalid training scalers")
    args.device = torch.device(device)
    model = Model(args).to(args.device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.freeze_ar()
    model.eval()
    return model, meta, fs, cs


def generate_raw(model, histories, targets, feature_scaler, condition_scaler, *,
                 paths_per_condition, batch_size, diffusion_seed):
    """Use the existing inference API with a zero future decoder, never real Y."""
    import torch
    if batch_size < 1 or paths_per_condition < 2:
        raise ValueError("Need batch_size >= 1 and paths_per_condition >= 2 for diversity checks")
    torch.manual_seed(diffusion_seed)
    h = feature_scaler.transform(histories)
    c = condition_scaler.transform(targets)
    draw_ids = np.repeat(np.arange(len(histories)), paths_per_condition)
    raw = np.empty((len(draw_ids), PRED_LEN, N_VARS), dtype=np.float32)
    started = time.monotonic()
    with torch.inference_mode():
        for b in range(0, len(draw_ids), batch_size):
            ids = draw_ids[b:b + batch_size]
            history = torch.as_tensor(h[ids], device=model.device)
            condition = torch.as_tensor(c[ids], device=model.device)
            result = model(history, None, decoder(history), None, sample_times=1,
                           market_condition=condition)[0][:, 0]
            raw[b:b + len(ids)] = feature_scaler.inverse(result.cpu().numpy())
            if not np.isfinite(raw[b:b + len(ids)]).all():
                raise RuntimeError("Non-finite generated features; no invalid paths are silently removed")
            if b == 0 or (b // batch_size + 1) % 10 == 0 or b + len(ids) == len(draw_ids):
                print(f"Generated {b + len(ids)}/{len(draw_ids)} paths in {time.monotonic() - started:.1f}s", flush=True)
    return raw, draw_ids


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "generate"):
        q = sub.add_parser(name)
        q.add_argument("--data", type=Path, default=ROOT / "datasets/dow10/prices_2004.csv")
        q.add_argument("--out", type=Path, required=True, help="Must not already exist")
        q.add_argument("--train-start", default="2010-01-01")
        q.add_argument("--train-end", default="2017-12-31")
        q.add_argument("--n-draws", type=int, default=500)
        q.add_argument("--paths-per-condition", type=int, default=4)
        q.add_argument("--full-weight", type=float, default=.70)
        q.add_argument("--tail-quantile", type=float, default=.30)
        q.add_argument("--plan-seed", type=int, default=20261017)
        if name == "generate":
            q.add_argument("--checkpoint", type=Path, required=True)
            q.add_argument("--diffusion-seed", type=int, default=20261018)
            q.add_argument("--batch-size", type=int, default=32)
            q.add_argument("--threads", type=int, default=4)
            q.add_argument("--device", default="cpu")
    return parser


def run(cli):
    started = time.monotonic()
    if cli.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {cli.out}")
    if cli.paths_per_condition < 2:
        raise ValueError("Need at least 2 paths per condition to measure diversity")
    prices, features = load_features(cli.data)
    plan = make_joint_plan(features, train_start=cli.train_start, train_end=cli.train_end,
                           n_draws=cli.n_draws, full_weight=cli.full_weight,
                           tail_quantile=cli.tail_quantile, seed=cli.plan_seed)
    histories, references, targets = paired_inputs(features, plan.draws)
    provenance = source_metadata(cli.data)
    manifest = {
        "format_version": 1, "status": "condition_plan_only", "ppo_run": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
        "sampling": plan.metadata, "paths_per_condition": cli.paths_per_condition,
        "n_planned_paths": len(plan.draws) * cli.paths_per_condition, **provenance,
    }
    if cli.command == "generate":
        import torch
        if cli.threads < 1 or cli.batch_size < 1:
            raise ValueError("threads and batch_size must be positive")
        torch.set_num_threads(cli.threads)
        model, meta, fs, cs = load_conditioned_model(
            cli.checkpoint, device=cli.device, train_start=cli.train_start,
            train_end=cli.train_end, data_sha256=provenance["data_sha256"])
        manifest.update({"status": "generating", "checkpoint": str(cli.checkpoint.resolve()),
                         "checkpoint_sha256": sha256(cli.checkpoint), "generator_metadata": meta,
                         "scenario_labels": {"0": "full source pool (may include tail outcomes)", "1": "lower-tail source pool"},
                         "diffusion_rng": "torch global stream; exact replay requires identical device, versions and batch size"})
        manifest["versions"]["torch"] = torch.__version__
    # Atomic creation also protects against concurrent runs selecting the same path.
    cli.out.mkdir(parents=True, exist_ok=False)
    plan.population.to_csv(cli.out / "population.csv", index=False)
    plan.draws.to_csv(cli.out / "sampling_plan.csv", index=False)
    def save_manifest():
        (cli.out / "generation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    save_manifest()
    if cli.command == "plan":
        print(json.dumps(manifest["sampling"], indent=2), flush=True)
        return manifest
    try:
        raw, ids = generate_raw(model, histories, targets, fs, cs,
                               paths_per_condition=cli.paths_per_condition, batch_size=cli.batch_size,
                               diffusion_seed=cli.diffusion_seed)
        path_meta = plan.draws.iloc[ids].reset_index(drop=True)
        path_meta.insert(0, "path_id", np.arange(len(ids)))
        path_meta["replicate_id"] = np.tile(np.arange(cli.paths_per_condition), len(plan.draws))
        anchor_close = wide_prices(prices)["close"].loc[pd.to_datetime(path_meta.anchor_date)].to_numpy()
        close, high, low, hclip, lclip = reconstruct_prices(
            anchor_close, raw, CLOSE_INDICES, [i + 1 for i in CLOSE_INDICES], [i + 2 for i in CLOSE_INDICES])
        close, high, low = (x.astype(np.float32) for x in (close, high, low))
        if not all(np.isfinite(x).all() and (x > 0).all() for x in (close, high, low)):
            raise RuntimeError("Generated prices must be finite and strictly positive")
        if not ((low <= close).all() and (close <= high).all()):
            raise RuntimeError("Invalid OHLC ordering")
        realized = portfolio_statistics(raw[..., CLOSE_INDICES])
        np.savez_compressed(
            cli.out / "paths.npz", tickers=np.array(TICKERS),
            anchor_dates=path_meta.anchor_date.to_numpy(dtype=str),
            # Compatibility field only: 0 = full-source draw, 1 = tail-source draw.
            scenario=(path_meta.source_pool == "tail").to_numpy(dtype=np.int64),
            source_pool=path_meta.source_pool.to_numpy(dtype=str),
            window_id=path_meta.window_id.to_numpy(), draw_id=path_meta.draw_id.to_numpy(),
            replicate_id=path_meta.replicate_id.to_numpy(),
            target=targets[ids], realized=realized,
            close=close, high=high, low=low,
            raw_features=raw, anchor_close=anchor_close,
            # These are audit-only; they must not become PPO observations.
            history_features=histories.astype(np.float32), history_draw_ids=plan.draws.draw_id.to_numpy(),
            paired_real_future=references.astype(np.float32),
        )
        from evaluation.joint_condition_diagnostics import write_diagnostics
        summary = write_diagnostics(cli.out, plan, raw, references[ids], path_meta,
                                    fs.scale, cs.scale, hclip, lclip)
        if sha256(cli.data) != provenance["data_sha256"] or sha256(cli.checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError("Source data or checkpoint changed while generating")
        manifest.update({"status": "complete", "n_generated_paths": len(raw),
                         "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.monotonic() - started,
                         "artifacts_sha256": {p.name: sha256(p) for p in sorted(cli.out.iterdir()) if p.is_file() and p.name != "generation.json"}})
        save_manifest()
        print(json.dumps({"out": str(cli.out), "n_paths": len(raw),
                          "target_tail_share": summary["target_tail_share"],
                          "realized_tail_share": summary["realized_tail_share"],
                          "diversity": summary["diversity"]}, indent=2), flush=True)
        return manifest
    except Exception as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        save_manifest()
        raise


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
