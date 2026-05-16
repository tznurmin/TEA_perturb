# Copyright 2026 tznurmin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pytest

from run_seed_sweep import (  # noqa: E402
    _compare_run_entries,
    _parse_methods,
    _selected_baselines,
)


def test_compare_run_entries_follow_selected_methods(tmp_path: Path) -> None:
    root = tmp_path / "summaries"

    assert _compare_run_entries(root, _parse_methods("none")) == [
        f"none={root / 'none'}"
    ]

    assert _compare_run_entries(root, _parse_methods("none,abtt,whiten")) == [
        f"none={root / 'none'}",
        f"abtt={root / 'abtt'}",
        f"white={root / 'whiten'}",
    ]


def test_no_rand_selects_synonym_baseline_only() -> None:
    assert _selected_baselines("both", no_rand=True) == ["synonym"]
    assert _selected_baselines("synonym", no_rand=True) == ["synonym"]

    with pytest.raises(SystemExit, match="incompatible"):
        _selected_baselines("random", no_rand=True)


def test_baseline_selection_includes_random_when_enabled() -> None:
    assert _selected_baselines("both", no_rand=False) == ["synonym", "random"]
    assert _selected_baselines("random", no_rand=False) == ["random"]
