from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _tokenizer_is_effectively_cased(tok) -> bool:
    # A cased tokeniser must distinguish between A and a
    try:
        ids_A = tok("A", add_special_tokens=True)["input_ids"]
        ids_a = tok("a", add_special_tokens=True)["input_ids"]
        return ids_A != ids_a
    except Exception:
        return False


def load_tokenizer(
    model_id: str,
    *,
    model_max_length: int = 512,
    use_fast: bool = False,
    enforce_cased: bool = True,
):
    """
    Tokeniser factory
    - Forces do_lower_case=False
    - Defaults to use_fast=False for compatibility
    - Cased behaviour via a behavioural check (optional)
    """
    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=use_fast,
        do_lower_case=False,
        model_max_length=model_max_length,
    )

    if enforce_cased and (("cased" in model_id) or ("biobert" in model_id.lower())):
        if not _tokenizer_is_effectively_cased(tok):
            raise RuntimeError(
                "Tokeniser is not behaving as cased (case-variants map to identical input_ids). "
                "This invalidates cased BioBERT assumptions. "
                "Fix: load with do_lower_case=False and use_fast=False"
            )

    return tok


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def load_model(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

    # Canonical tokeniser loading with cased enforcement
    # BioBERT v1.2 cased variant has a known failure mode where tokenisation becomes uncased unless forced
    tok = load_tokenizer(
        model_name,
        model_max_length=512,
        use_fast=False,
        enforce_cased=True,
    )

    model = AutoModel.from_pretrained(model_name, config=cfg).to(device).eval()
    return model, tok


@torch.inference_mode()
def encode_sentences(
    model: AutoModel,
    tokenizer,
    texts: List[str],
    batch_size: int = 32,
    pooling: str = "mean_last2",
) -> torch.Tensor:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=160, return_tensors="pt"
        ).to(device)
        out = model(**enc)
        if pooling == "cls":
            pooled = out.last_hidden_state[:, 0]
        elif pooling == "mean_last2":
            last2 = (out.hidden_states[-1] + out.hidden_states[-2]) / 2.0
            mask = enc["attention_mask"].unsqueeze(-1).expand(last2.size()).float()
            pooled = (last2 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            h = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(h.size()).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embs.append(pooled.detach().cpu())
    return torch.vstack(embs)


@torch.inference_mode()
def pairwise_compare(z1: torch.Tensor, z2: torch.Tensor):
    cos = F.cosine_similarity(z1, z2).cpu().numpy()
    z1u, z2u = F.normalize(z1, p=2, dim=1), F.normalize(z2, p=2, dim=1)
    l2_unit = (z1u - z2u).norm(dim=1).cpu().numpy()
    return cos, l2_unit
