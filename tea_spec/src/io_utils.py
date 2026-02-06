from __future__ import annotations
import json, re
from typing import List

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
