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
