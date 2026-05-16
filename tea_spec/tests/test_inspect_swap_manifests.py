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

from inspect_swap_manifests import _iter_manifest_paths  # noqa: E402


def test_manifest_discovery_searches_model_scoped_cache(tmp_path: Path) -> None:
    root = tmp_path / "work" / "cache"
    model_dir = root / "models" / "dmis-lab_biobert-base-cased-v1_2"
    model_dir.mkdir(parents=True)
    syn = model_dir / "species_embeddings_seed42.synswap.jsonl"
    rnd = model_dir / "species_embeddings_seed42.randswap.jsonl"
    syn.write_text("{}\n", encoding="utf-8")
    rnd.write_text("{}\n", encoding="utf-8")

    assert _iter_manifest_paths(root) == [syn, rnd]
