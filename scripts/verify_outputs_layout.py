"""Read-only relocation audit. Never train, generate, backtest, or run PPO."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from output_layout import MIGRATION, PR30_BATCH, PR10_BATCH, LEGACY_CONDITIONAL, sha256, verify_historical_hash, resolve_output_path
from evaluation.loaders import discover_models, load_synthetic


def main():
    inventory = json.loads((MIGRATION / "inventory.json").read_text())
    unchanged, edited = [], []
    for row in inventory["files"]:
        path = ROOT / row["new"]
        status = verify_historical_hash(path, row["sha256"])
        (unchanged if status == "unchanged" else edited).append(row["new"])
        if path.suffix not in {".py", ".md", ".ipynb", ".sh"}:
            assert status == "unchanged", path
    # Existing notebook outputs are historical results, not disposable caches.
    for row in inventory["source_snapshots"]:
        if not row["path"].endswith(".ipynb"):
            continue
        old = json.loads((MIGRATION / "original_sources" / row["repository"] / row["path"]).read_text())
        now_path = ROOT.parent / row["repository"] / row["path"]
        if row["repository"] == "TimeDiff":
            now_path = resolve_output_path(now_path)
        new = json.loads(now_path.read_text())
        assert len(old["cells"]) == len(new["cells"])
        for a, b in zip(old["cells"], new["cells"]):
            assert a.get("outputs") == b.get("outputs"), now_path
            assert a.get("execution_count") == b.get("execution_count"), now_path
    arrays = []
    for row in inventory["files"]:
        path = ROOT / row["new"]
        if path.suffix != ".npz":
            continue
        with np.load(path, allow_pickle=False) as z:
            shapes = {k: list(z[k].shape) for k in z.files}
        arrays.append({"path": row["new"], "arrays": shapes})
    batches = []
    for label, path, n in [("fixed_scenarios", LEGACY_CONDITIONAL / "rl_paths", 6039),
                           ("PR30", PR30_BATCH, 2000), ("PR10", PR10_BATCH, 2000)]:
        with np.load(path / "paths.npz", allow_pickle=False) as z:
            assert z["close"].shape == (n, 21, 10)
            assert np.isfinite(z["raw_features"]).all()
            assert (z["low"] <= z["close"]).all() and (z["close"] <= z["high"]).all()
            batches.append({"batch": label, "n_paths": n, "shape": list(z["close"].shape), "sha256": sha256(path / "paths.npz")})
    models, reads, issues = discover_models(verbose=False), [], []
    for model in models:
        details = []
        for f in model["files"]:
            frame = pd.read_csv(f)
            details.append({"file": Path(f).name, "rows": len(frame), "first_date": str(frame.date.iloc[0]), "last_date": str(frame.date.iloc[-1])})
        try:
            dates, values = load_synthetic(model)
            reads.append({"tag": model["tag"], "status": "loaded", "n_files": len(details), "shape": list(values.shape), "files": details})
        except (ValueError, KeyError) as error:
            # These bytes were verified unchanged above: this is a pre-existing issue.
            issues.append({"tag": model["tag"], "status": "preexisting_data_issue", "error": str(error), "files": details})
    report = {"status": "passed_with_preexisting_data_issues" if issues else "passed",
              "original_files": len(inventory["files"]), "byte_identical_files": len(unchanged),
              "source_or_doc_updates_with_original_backup": edited,
              "notebook_saved_outputs_preserved": True, "npz_archives_loaded": len(arrays),
              "npz_arrays": arrays, "conditional_batches": batches,
              "discovered_models": len(models), "loaded_models": reads, "preexisting_issues": issues,
              "training_run": False, "generation_run": False, "ppo_run": False}
    (MIGRATION / "readability_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "original_files": len(inventory["files"]),
        "byte_identical_files": len(unchanged), "preserved_source_updates": len(edited),
        "npz_archives_loaded": len(arrays), "discovered_models": len(models),
        "models_loaded": [m["tag"] for m in reads],
        "preexisting_data_issues": [m["tag"] for m in issues]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
