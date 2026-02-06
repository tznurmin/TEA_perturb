from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ensure we can import the src modules (scripts are flat modules in tea_spec/src)
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from setup_workdir import (ensure_scaffold,  # noqa: E402
                           validate_tea_curated_data)


def test_scaffold_creation(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    ensure_scaffold(workdir)

    expected = [
        workdir / "TEA_curated_data",
        workdir / "species",
        workdir / "templates",
        workdir / "cache",
        workdir / "summaries_from_cache",
        workdir / "_downloads",
    ]
    for p in expected:
        assert p.exists() and p.is_dir()


def test_validation_fails_on_empty_dataset_dir(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    ensure_scaffold(workdir)

    tea_root = workdir / "TEA_curated_data"
    res = validate_tea_curated_data(tea_root)

    assert not res.ok
    # must list at least one concrete missing path or content issue
    assert len(res.issues) > 0
