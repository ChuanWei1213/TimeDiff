"""One-time, hash-checked output relocation. Never train or generate data.

Run --inventory before any edits, then --apply. Original manifests stay intact.
The inventory also keeps source snapshots used to verify historical code hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/_migration/20261018"
RUN = "outputs/dow10/conditional/empirical_joint/run_20261017"
GROUPS = {
    "timediff": ["dow10_wf_v2", "dow10_wf_vpred", "v_cross", "v_cross_ga", "v_whiten", "v_3layer", "v_4layer", "v_4layer_fix"],
    "statistical": ["gauss_sigma", "garch_null"],
    "external": ["armd", "cndiff", "nsdiff", "dva"],
}
MAPPING = {f"outputs/{tag}": f"outputs/dow10/walk_forward/{group}/{tag}" for group, tags in GROUPS.items() for tag in tags}
MAPPING.update({
    "outputs/dow10_cond": "outputs/dow10/conditional/fixed_scenarios/legacy",
    "outputs/dow10_joint_20261017/batch": RUN + "/batches/full70_tail30_pr30",
    "outputs/dow10_joint_20261017/condition_plan": RUN + "/plans/full70_tail30_pr30",
    "outputs/dow10_joint_20261017": RUN,
    "outputs/dow10_joint_pr10_20261018/batch": RUN + "/batches/full70_tail30_pr10",
    "outputs/dow10_joint_pr10_20261018": RUN + "/analyses/pr30_vs_pr10",
    "outputs/dow10_joint_diagnosis_20261017": RUN + "/analyses/generation_quality",
    "outputs/dow10_stock_tail_design_20261018": RUN + "/designs/random_stock_pr30",
    "outputs/p0p1_results.txt": "outputs/dow10/walk_forward/analyses/p0p1_results.txt",
})


def mapped(path):
    path = str(path)
    for old, new in sorted(MAPPING.items(), key=lambda pair: -len(pair[0])):
        if path == old or path.startswith(old + "/"):
            return new + path[len(old):]
    return path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def inventory():
    if AUDIT.exists():
        raise FileExistsError(f"Refusing to replace migration evidence: {AUDIT}")
    files = []
    for p in sorted((ROOT / "outputs").rglob("*")):
        if p.is_symlink():
            raise ValueError(f"Review symlink before migration: {p}")
        if p.is_file():
            rel = p.relative_to(ROOT).as_posix()
            files.append({"old": rel, "new": mapped(rel), "size": p.stat().st_size, "sha256": sha(p)})
    AUDIT.mkdir(parents=True)
    # Snapshot editable source/docs before path updates. Preserve notebook outputs too.
    snapshots = []
    for repo in (ROOT, ROOT.parent / "FinRL_Exp"):
        for p in repo.rglob("*"):
            rel = p.relative_to(repo)
            if any(part.startswith((".git", ".venv")) or part in {"venv", "node_modules", "__pycache__", "_migration", "datasets", "RL"} for part in rel.parts):
                continue
            if not p.is_file() or p.suffix not in {".py", ".md", ".ipynb", ".sh"}:
                continue
            dest = AUDIT / "original_sources" / repo.name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            snapshots.append({"repository": repo.name, "path": rel.as_posix(), "sha256": sha(p)})
    shutil.copy2(ROOT / ".gitignore", AUDIT / "original_sources/TimeDiff/.gitignore")
    payload = {"version": 1, "status": "inventoried", "mapping": MAPPING, "files": files,
               "source_snapshots": snapshots, "training_run": False, "ppo_run": False}
    (AUDIT / "inventory.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"files": len(files), "bytes": sum(f["size"] for f in files), "source_snapshots": len(snapshots)}))


def apply():
    record = json.loads((AUDIT / "inventory.json").read_text())
    # Validate every original before moving; allow a verified interrupted move.
    for row in record["files"]:
        source, destination = ROOT / row["old"], ROOT / row["new"]
        if source.exists():
            assert sha(source) == row["sha256"], row["old"]
            if source != destination and destination.exists():
                raise FileExistsError(row["new"])
        else:
            assert sha(destination) == row["sha256"], row["new"]
    for row in record["files"]:
        if row["old"] == row["new"]:
            continue
        if not (ROOT / row["old"]).exists():
            continue
        dest = ROOT / row["new"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        (ROOT / row["old"]).rename(dest)
    # Preserve emptied directory trees too; this host does not permit rmdir.
    empty_directories_retained = []
    for old in sorted({"/".join(p.split("/")[:2]) for p in MAPPING}):
        p = ROOT / old
        if p.is_dir():
            assert not any(q.is_file() or q.is_symlink() for q in p.rglob("*")), old
            destination = AUDIT / "empty_legacy_directories" / p.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            try:
                p.rename(destination)
            except PermissionError:
                empty_directories_retained.append(old)
    for row in record["files"]:
        assert sha(ROOT / row["new"]) == row["sha256"], row["new"]
    (AUDIT / "relocation.json").write_text(json.dumps({"status": "relocated", "files_verified": len(record["files"]),
        "all_original_bytes_preserved": True, "mapping": MAPPING,
        "empty_legacy_directories_retained_by_host": empty_directories_retained}, indent=2) + "\n")
    print(f"Relocated and verified {len(record['files'])} original files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--inventory", action="store_true")
    action.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    inventory() if args.inventory else apply()
