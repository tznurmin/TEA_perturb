from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def pairwise_compare(z1: torch.Tensor, z2: torch.Tensor):
    """Compute per-row cosine similarity and unit-normalized L2 distance.

    Returns:
      cos: numpy array (N,)
      l2_unit: numpy array (N,)
    """
    cos = F.cosine_similarity(z1, z2)
    z1u, z2u = F.normalize(z1, p=2, dim=1), F.normalize(z2, p=2, dim=1)
    l2_unit = (z1u - z2u).norm(dim=1)
    return cos.detach().cpu().numpy(), l2_unit.detach().cpu().numpy()
