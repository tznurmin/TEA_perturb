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
import json, re
from typing import List


def normalize_model_id(model: str) -> str:
    """Normalize a model identifier.

    Supported formats:
    - "<hf_repo_id>" (assumed Hugging Face)
    - "hf:<hf_repo_id>" (optional vendor prefix)
    """
    s = (model or "").strip()
    if s.lower().startswith("hf:"):
        s = s[3:].strip()
    return s


def slugify_model_id(model: str) -> str:
    """Convert a model identifier into a filesystem-safe slug."""
    s = normalize_model_id(model)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "model"


def read_species_list(path: str) -> List[str]:
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            names.append(s)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def load_placeholder_templates(dataset_path: str, placeholder: str, require_single: bool) -> List[str]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        text = r["text"]
        c = text.count(placeholder)
        if require_single and c == 1:
            out.append(text)
        elif not require_single and c >= 1:
            out.append(text)
    return out
