"""Canonical output paths and explicit compatibility for archived run manifests."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
JOINT_RUN = OUTPUTS / "dow10/conditional/empirical_joint/run_20261017"
PR30_BATCH = JOINT_RUN / "batches/full70_tail30_pr30"
PR10_BATCH = JOINT_RUN / "batches/full70_tail30_pr10"
LEGACY_CONDITIONAL = OUTPUTS / "dow10/conditional/fixed_scenarios/legacy"
MIGRATION = OUTPUTS / "_migration/20261018"
GROUPS = {
    "timediff": ("dow10_wf_v2", "dow10_wf_vpred", "v_cross", "v_cross_ga", "v_whiten", "v_3layer", "v_4layer", "v_4layer_fix"),
    "statistical": ("gauss_sigma", "garch_null"),
    "external": ("armd", "cndiff", "nsdiff", "dva"),
}
LEGACY_PREFIXES = {f"outputs/{tag}": f"outputs/dow10/walk_forward/{group}/{tag}" for group, tags in GROUPS.items() for tag in tags}
_RUN = JOINT_RUN.relative_to(ROOT).as_posix()
LEGACY_PREFIXES.update({
    "outputs/dow10_cond": LEGACY_CONDITIONAL.relative_to(ROOT).as_posix(),
    "outputs/dow10_joint_20261017/batch": _RUN + "/batches/full70_tail30_pr30",
    "outputs/dow10_joint_20261017/condition_plan": _RUN + "/plans/full70_tail30_pr30",
    "outputs/dow10_joint_20261017": _RUN,
    "outputs/dow10_joint_pr10_20261018/batch": _RUN + "/batches/full70_tail30_pr10",
    "outputs/dow10_joint_pr10_20261018": _RUN + "/analyses/pr30_vs_pr10",
    "outputs/dow10_joint_diagnosis_20261017": _RUN + "/analyses/generation_quality",
    "outputs/dow10_stock_tail_design_20261018": _RUN + "/designs/random_stock_pr30",
    "outputs/p0p1_results.txt": "outputs/dow10/walk_forward/analyses/p0p1_results.txt",
})


def walkforward_path(tag, *, group=None, outputs_root=OUTPUTS):
    """Known tags retain their group; new native tags default to timediff."""
    if not tag or Path(tag).name != tag or tag in {".", ".."}:
        raise ValueError("tag must be one directory name")
    inferred = next((g for g, tags in GROUPS.items() if tag in tags), None)
    group = inferred or group or "timediff"
    if group not in GROUPS:
        raise ValueError(f"Unknown walk-forward group: {group}")
    return Path(outputs_root) / "dow10/walk_forward" / group / tag


def resolve_output_path(path):
    """Resolve historical output paths without changing immutable run metadata."""
    path = Path(path).expanduser()
    if not path.is_absolute() and path.parts and path.parts[0] == "outputs":
        path = ROOT / path
    absolute = path.absolute()
    # Normalize .. but never require the old file to exist.
    absolute = absolute.resolve(strict=False)
    try:
        relative = absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute
    for old, new in sorted(LEGACY_PREFIXES.items(), key=lambda item: -len(item[0])):
        if relative == old or relative.startswith(old + "/"):
            return ROOT / (new + relative[len(old):])
    return absolute


def model_directories(outputs_root=OUTPUTS):
    """Discover classified models plus legacy flat/custom evaluation fixtures."""
    root = Path(outputs_root)
    candidates = list(root.glob("*"))
    for group in GROUPS:
        candidates.extend((root / "dow10/walk_forward" / group).glob("*"))
    candidates.extend((root / "dow10/single_shot").glob("*"))
    return sorted({p for p in candidates if p.is_dir()}, key=lambda p: p.as_posix())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_historical_hash(path, expected):
    """Verify original bytes, or an explicitly recorded relocation-only source edit.

    Edited docs/code must match both the preserved original and reviewed new hash.
    Unrecorded subsequent edits still fail. Generated arrays/models are never exempt.
    """
    path = resolve_output_path(path)
    current = sha256(path)
    if current == expected:
        return "unchanged"
    changes_file = MIGRATION / "source_updates.json"
    if changes_file.exists():
        for row in json.loads(changes_file.read_text())["files"]:
            if Path(row["current_absolute"]) == path and row["old_sha256"] == expected and row["new_sha256"] == current:
                original = MIGRATION / row["backup"]
                if sha256(original) == expected:
                    return "preserved_original_and_verified_path_update"
    raise ValueError(f"Historical hash mismatch: {path}")
