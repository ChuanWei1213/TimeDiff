"""Path relocation must not change model identity or archived-config meaning."""
from pathlib import Path
import json

import pandas as pd
import pytest

import output_layout as layout
from evaluation.loaders import discover_models, load_synthetic


def test_archived_joint_paths_use_most_specific_prefix():
    assert layout.resolve_output_path("outputs/dow10_joint_20261017/batch/paths.npz") == layout.PR30_BATCH / "paths.npz"
    assert layout.resolve_output_path(layout.ROOT / "outputs/dow10_joint_pr10_20261018/batch/paths.npz") == layout.PR10_BATCH / "paths.npz"
    assert layout.resolve_output_path("outputs/dow10_joint_20261017/train/model.pt") == layout.JOINT_RUN / "train/model.pt"
    assert layout.resolve_output_path("outputs/dow10_joint_20261017_other/paths.npz") == layout.OUTPUTS / "dow10_joint_20261017_other/paths.npz"


def test_model_groups_and_unknown_generation_tags():
    assert layout.walkforward_path("armd").parent.name == "external"
    assert layout.walkforward_path("garch_null").parent.name == "statistical"
    assert layout.walkforward_path("new_model").parent.name == "timediff"
    assert layout.walkforward_path("new_import", group="external").parent.name == "external"
    with pytest.raises(ValueError):
        layout.walkforward_path("../escape")


def test_discovery_reads_all_groups_and_legacy_fixture(tmp_path):
    expected = {"dow10_wf_vpred", "gauss_sigma", "armd", "legacy_custom"}
    for tag in expected:
        root = tmp_path / tag if tag == "legacy_custom" else layout.walkforward_path(tag, outputs_root=tmp_path)
        directory = root / "paths/returns"
        directory.mkdir(parents=True)
        frame = pd.DataFrame({"date": ["2018-01-02", "2018-01-03"],
                              **{f"{ticker}_rclose": [.01, -.02] for ticker in ("AAPL", "AMGN", "CRM", "CSCO", "IBM", "INTC", "MSFT", "NKE", "VZ", "WMT")}})
        frame.to_csv(directory / "path_00_returns.csv", index=False)
    models = discover_models(tmp_path, verbose=False)
    assert {m["tag"] for m in models} == expected
    for model in models:
        dates, values = load_synthetic(model)
        assert values.shape == (1, 2, 10)
        assert str(dates[0].date()) == "2018-01-02"


def test_historical_hash_requires_current_and_original_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "MIGRATION", tmp_path)
    current, backup = tmp_path / "source.py", tmp_path / "original.py"
    backup.write_text("old path")
    old_hash = layout.sha256(backup)
    current.write_text("new path")
    record = {"files": [{"current_absolute": str(current), "backup": backup.name,
                          "old_sha256": old_hash, "new_sha256": layout.sha256(current)}]}
    (tmp_path / "source_updates.json").write_text(json.dumps(record))
    assert layout.verify_historical_hash(current, old_hash) == "preserved_original_and_verified_path_update"
    current.write_text("unreviewed later edit")
    with pytest.raises(ValueError):
        layout.verify_historical_hash(current, old_hash)
