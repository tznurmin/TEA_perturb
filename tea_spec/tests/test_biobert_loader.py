# tea_spec/tests/test_biobert_loader.py

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _add_src_to_syspath() -> None:
    # .../tea_spec/tests -> .../tea_spec/src
    src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src))


_add_src_to_syspath()

import encode  # noqa: E402


class _DummyTokenizerAlwaysUncased:
    def __call__(self, text, add_special_tokens: bool = True, **kwargs):
        if isinstance(text, list) and text:
            text = text[0]
        t = str(text).lower()
        return {"input_ids": [101, hash(t) % 10000, 102]}


class _DummyTokenizerCased:
    def __call__(self, text, add_special_tokens: bool = True, **kwargs):
        if isinstance(text, list) and text:
            text = text[0]
        t = str(text)
        return {"input_ids": [101, hash(t) % 10000, 102]}


def test_load_model_raises_if_tokenizer_behaves_uncased(monkeypatch) -> None:
    # prevent HF downloads
    monkeypatch.setattr(encode.AutoConfig, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(encode.AutoModel, "from_pretrained", lambda *a, **k: object())

    # patch the symbol used by encode.load_tokenizer()
    monkeypatch.setattr(
        encode.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: _DummyTokenizerAlwaysUncased(),
    )

    with pytest.raises(RuntimeError, match="not behaving as cased"):
        encode.load_model("dmis-lab/biobert-base-cased-v1.2")


def test_load_model_accepts_tokenizer_that_behaves_cased(monkeypatch) -> None:
    monkeypatch.setattr(encode.AutoConfig, "from_pretrained", lambda *a, **k: object())

    class _M:
        def to(self, *a, **k):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(encode.AutoModel, "from_pretrained", lambda *a, **k: _M())

    monkeypatch.setattr(
        encode.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: _DummyTokenizerCased(),
    )

    model, tok = encode.load_model("dmis-lab/biobert-base-cased-v1.2")
    assert tok("A")["input_ids"] != tok("a")["input_ids"]
