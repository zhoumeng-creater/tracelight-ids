"""Weighted attention with an explicit, already-scaled query contract."""

from __future__ import annotations

import math

from torch import Tensor
from torch.nn import functional as F

from .cache import WeightedKVEntry


def prepare_query(query: Tensor, source_count: int, ssmax_layer=None) -> Tensor:
    """Apply QASSMax to projected Q, then SDPA's 1/sqrt(d), exactly once.

    Query-aware SSMax is nonlinear: scaling Q before the module is not an
    equivalent transformation. ``source_count`` is N, even when M < N.
    """
    if query.ndim != 4 or query.shape[-1] < 1 or source_count < 1:
        raise ValueError("expected [B,H,Q,d] query and positive original source_count")
    transformed = query if ssmax_layer is None else ssmax_layer(query, source_count)
    return transformed / math.sqrt(query.shape[-1])


def weighted_attention(query_phi: Tensor, entry: WeightedKVEntry) -> Tensor:
    """Return softmax(phi_N(Q) K.T + log_mass) V.

    The input has already passed QASSMax and sqrt(d) scaling. Use scale=1 to
    avoid applying the SDPA scale twice. The additive mask broadcasts over Q.
    This same PyTorch SDPA route is used for full and compressed cache checks.
    """
    if (query_phi.ndim != 4 or query_phi.shape[:2] != entry.key.shape[:2]
            or query_phi.shape[-1] != entry.key.shape[-1]):
        raise ValueError("query must match the cache batch, heads and head dimension")
    if query_phi.dtype != entry.key.dtype or query_phi.device != entry.key.device:
        raise ValueError("query and cache must share dtype and device")
    return F.scaled_dot_product_attention(
        query_phi, entry.key, entry.value, attn_mask=entry.log_mass,
        dropout_p=0.0, is_causal=False, scale=1.0,
    )
