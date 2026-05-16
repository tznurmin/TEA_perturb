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

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# WordNet is used for synonym swaps
_wordnet_ready = False
try:
    from nltk.corpus import wordnet as wn  # type: ignore

    _ = wn.synsets("dog")
    _wordnet_ready = True
except Exception:
    wn = None  # type: ignore
    _wordnet_ready = False


def abbreviate_binomial(name: str) -> str:
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    return f"{parts[0][0]}. {parts[1]}"


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def bootstrap_ci(
    values, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    n = len(vals)
    idx = np.arange(n)
    boots = [vals[rng.choice(idx, size=n, replace=True)].mean() for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - level) / 2 * 100)
    hi = np.percentile(boots, (1 + level) / 2 * 100)
    return float(lo), float(hi)


def summarize(vec: np.ndarray, n_boot: int = 2000, level: float = 0.95) -> dict:
    return {
        "mean": float(np.mean(vec)) if vec.size else float("nan"),
        "ci95": tuple(map(float, bootstrap_ci(vec, n_boot=n_boot, level=level)))
        if vec.size
        else (float("nan"), float("nan")),
        "n": int(vec.size),
    }


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "at",
    "into",
    "over",
    "under",
    "within",
}

_PUNCT_STRIP = "\"'`.,;:!?()[]{}<>"


@dataclass(frozen=True)
class SynonymSwapResult:
    ok: bool
    text_out: Optional[str]
    reason: str
    attempts: int
    changed_token: Optional[str] = None
    replacement: Optional[str] = None


@dataclass(frozen=True)
class RandomSwapResult:
    ok: bool
    text_out: Optional[str]
    reason: str
    attempts: int
    changed_token: Optional[str] = None
    replacement: Optional[str] = None


def _stable_seed(text: str, protected: Sequence[str], seed: int) -> int:
    # derive a per-example deterministic seed
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    for p in protected:
        h.update(b"\0")
        h.update(p.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little", signed=False)


def _tokenize_for_diff(s: str) -> List[str]:
    toks: List[str] = []
    for raw in s.split():
        t = raw.strip(_PUNCT_STRIP)
        toks.append(t)
    return toks


def _diff_token_positions(a: str, b: str) -> int:
    ta = _tokenize_for_diff(a)
    tb = _tokenize_for_diff(b)
    if len(ta) != len(tb):
        return 999_999
    return sum(1 for x, y in zip(ta, tb) if x != y)


def _normalize_word(w: str) -> str:
    return re.sub(r"\s+", " ", w.replace("_", " ").strip().lower())


def _case_match(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst.capitalize()
    return dst.lower()


def _majority_pos(word: str) -> Optional[str]:
    counts = {"n": 0, "v": 0, "a": 0, "r": 0}
    for syn in wn.synsets(word):
        p = syn.pos()
        if p in counts:
            counts[p] += 1
    return max(counts, key=counts.get) if sum(counts.values()) else None


def _spans_for_phrases(s: str, phrases: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for phrase in phrases:
        if not phrase:
            continue
        start = 0
        while True:
            i = s.find(phrase, start)
            if i < 0:
                break
            spans.append((i, i + len(phrase)))
            start = i + len(phrase)
    return spans


def _overlaps(i: int, j: int, spans: Sequence[Tuple[int, int]]) -> bool:
    for a, b in spans:
        if not (j <= a or i >= b):
            return True
    return False


def make_synonym_swapper(
    protected_phrases: List[str],
    seed: int,
    max_attempts: int = 250,
    require_one_token_change: bool = True,
) -> "callable[[str], SynonymSwapResult]":

    """
    Return a deterministic synonym swapper with strict invariants.

    Safeguards
     - never edits protected phrases
     - must change exactly one token
     - must not return unchanged input
     - fail-closed with ok=False rather than silently returning identity

    Determinism: the chosen token + synonym are derived from a stable hash of (text, protected, seed).
    """

    prot = [p for p in (protected_phrases or []) if p]
    cache: Dict[str, SynonymSwapResult] = {}

    def _candidates(s: str) -> List[Tuple[re.Match[str], List[str]]]:
        spans = _spans_for_phrases(s, prot)
        out: List[Tuple[re.Match[str], List[str]]] = []

        for m in re.finditer(r"\b[A-Za-z]{3,}\b", s):
            i, j = m.span()
            if _overlaps(i, j, spans):
                continue

            w = m.group(0)
            wl = w.lower()

            # reject capitalized tokens (tends to hit sentence-initial words / proper nouns)
            if w[:1].isupper():
                continue

            if wl in _STOPWORDS:
                continue

            pos = _majority_pos(wl)
            synsets = wn.synsets(wl, pos=pos) if pos else wn.synsets(wl)

            # build a vetted synonym list
            syns = []
            seen = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ")
                    if " " in name:
                        continue
                    if not re.fullmatch(r"[A-Za-z]+", name):
                        continue
                    if _normalize_word(name) == _normalize_word(w):
                        continue
                    key = _normalize_word(name)
                    if key in seen:
                        continue
                    seen.add(key)
                    syns.append(name)

            if syns:
                out.append((m, syns))

        return out

    def _attempt_swap_once(s: str, attempt_seed: int) -> Optional[Tuple[str, str, str]]:
        # returns (out_text, changed_token, replacement)
        cands = _candidates(s)
        if not cands:
            return None

        # deterministic choice per input, not affected by evaluation order
        rng = np.random.default_rng(attempt_seed)

        # choose candidate index deterministically
        cand_idx = int(rng.integers(0, len(cands)))
        m, syns = cands[cand_idx]

        # choose synonym deterministically
        syn_idx = int(rng.integers(0, len(syns)))
        repl_raw = syns[syn_idx]

        src = m.group(0)
        repl = _case_match(src, repl_raw)

        out = s[: m.start()] + repl + s[m.end() :]
        return out, src, repl

    def swap(text: str) -> SynonymSwapResult:
        if text in cache:
            return cache[text]

        # if WordNet is unavailable, fail closed with an explicit reason
        if not _wordnet_ready:
            res = SynonymSwapResult(
                ok=False, text_out=None, reason="wordnet_unavailable", attempts=0
            )
            cache[text] = res
            return res

        base_seed = _stable_seed(text, prot, seed)

        for attempt in range(1, max_attempts + 1):
            attempt_seed = base_seed + attempt
            triple = _attempt_swap_once(text, attempt_seed)
            if triple is None:
                res = SynonymSwapResult(
                    ok=False, text_out=None, reason="no_candidate", attempts=attempt
                )
                cache[text] = res
                return res

            out, changed, repl = triple

            if out == text:
                continue

            # protected phrase integrity: must preserve exact substrings and counts
            protected_ok = True
            for phrase in prot:
                if phrase and out.count(phrase) != text.count(phrase):
                    protected_ok = False
                    break
            if not protected_ok:
                continue

            if require_one_token_change:
                if _diff_token_positions(text, out) != 1:
                    continue

            # must be a real replacement
            if _normalize_word(changed) == _normalize_word(repl):
                continue

            res = SynonymSwapResult(
                ok=True,
                text_out=out,
                reason="ok",
                attempts=attempt,
                changed_token=changed,
                replacement=repl,
            )
            cache[text] = res
            return res

        res = SynonymSwapResult(
            ok=False,
            text_out=None,
            reason="max_attempts_exhausted",
            attempts=max_attempts,
        )
        cache[text] = res
        return res

    return swap


def make_synonym_swap_fn(
    protected_phrases: List[str], seed: int, max_attempts: int = 250
):

    swapper = make_synonym_swapper(
        protected_phrases=protected_phrases, seed=seed, max_attempts=max_attempts
    )

    def fn(x: str) -> str:
        res = swapper(x)
        if not res.ok or res.text_out is None:
            raise ValueError(f"Synonym swap unsuccessful: {res.reason}")
        return res.text_out

    return fn


def wordnet_is_ready() -> bool:
    return _wordnet_ready


_wordfreq_ready = False
_wordfreq_words: List[str] = []
try:
    from wordfreq import top_n_list  # type: ignore

    # a large list makes collisions (replacement==source) extremely unlikely
    # filter to plain alphabetic tokens, keep a single-token replacement invariant
    _raw = top_n_list("en", 50_000)
    _wordfreq_words = [
        w
        for w in _raw
        if re.fullmatch(r"[a-z]+", w) and len(w) >= 3 and w not in _STOPWORDS
    ]
    _wordfreq_ready = len(_wordfreq_words) > 0
except Exception:
    _wordfreq_ready = False
    _wordfreq_words = []


def wordfreq_is_ready() -> bool:
    return _wordfreq_ready


def make_random_word_swapper(
    protected_phrases: List[str],
    seed: int,
    max_attempts: int = 250,
    wordlist_size: int = 50_000,
    require_one_token_change: bool = True,
) -> "callable[[str], RandomSwapResult]":

    """
    Return a deterministic one-token swapper using wordfreq top English words.

    This is a meaning-breaking control baseline, intended to measure embedding movement from an arbitrary lexical substitution rather than species identity.

    Safeguards:
      - never edits protected phrases
      - must change exactly one token (configurable)
      - must not return unchanged input
      - fail-closed with ok=False rather than silently returning identity

    Determinism: the chosen token + replacement are derived from a stable hash of (text, protected, seed).
    """

    prot = [p for p in (protected_phrases or []) if p]
    cache: Dict[str, RandomSwapResult] = {}

    # clamp size against the available list
    wl = _wordfreq_words[: max(0, min(int(wordlist_size), len(_wordfreq_words)))]

    def _candidates(s: str) -> List[re.Match[str]]:
        spans = _spans_for_phrases(s, prot)
        out: List[re.Match[str]] = []

        for m in re.finditer(r"\b[A-Za-z]{3,}\b", s):
            i, j = m.span()
            if _overlaps(i, j, spans):
                continue

            w = m.group(0)
            wl0 = w.lower()

            # reject capitalized tokens (tends to hit sentence-initial words / proper nouns)
            if w[:1].isupper():
                continue

            if wl0 in _STOPWORDS:
                continue

            out.append(m)

        return out

    def swap(text: str) -> RandomSwapResult:
        if text in cache:
            return cache[text]

        if not _wordfreq_ready or not wl:
            res = RandomSwapResult(
                ok=False, text_out=None, reason="wordfreq_unavailable", attempts=0
            )
            cache[text] = res
            return res

        base_seed = _stable_seed(text, prot, seed)

        for attempt in range(1, max_attempts + 1):
            attempt_seed = base_seed + attempt
            rng = np.random.default_rng(attempt_seed)

            cands = _candidates(text)
            if not cands:
                res = RandomSwapResult(
                    ok=False, text_out=None, reason="no_candidate", attempts=attempt
                )
                cache[text] = res
                return res

            m = cands[int(rng.integers(0, len(cands)))]
            src = m.group(0)

            repl_raw = wl[int(rng.integers(0, len(wl)))]
            repl = _case_match(src, repl_raw)

            # must be a real replacement
            if _normalize_word(src) == _normalize_word(repl):
                continue

            out = text[: m.start()] + repl + text[m.end() :]

            if out == text:
                continue

            # protected phrase integrity: must preserve exact substrings and counts
            protected_ok = True
            for phrase in prot:
                if phrase and out.count(phrase) != text.count(phrase):
                    protected_ok = False
                    break
            if not protected_ok:
                continue

            if require_one_token_change:
                if _diff_token_positions(text, out) != 1:
                    continue

            res = RandomSwapResult(
                ok=True,
                text_out=out,
                reason="ok",
                attempts=attempt,
                changed_token=src,
                replacement=repl,
            )
            cache[text] = res
            return res

        res = RandomSwapResult(
            ok=False,
            text_out=None,
            reason="max_attempts_exhausted",
            attempts=max_attempts,
        )
        cache[text] = res
        return res

    return swap
