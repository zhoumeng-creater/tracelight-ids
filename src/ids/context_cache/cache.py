"""Typed weighted Stage 3 cache; original context length is independent of M."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from . import ARTIFACT_VERSION


@dataclass
class WeightedKVEntry:
    """K/V [batch, heads, slots, dim], log_mass [batch, heads, 1, slots].

    The first adapter supports float32/float64, with the same dtype and device
    for all fields. Counts sum to ``source_count`` for every batch/head.
    No training rows or calibration queries are held in this object.
    """

    key: Tensor
    value: Tensor
    log_mass: Tensor
    source_count: int

    def __post_init__(self):
        if isinstance(self.source_count, bool) or not isinstance(self.source_count, int):
            raise ValueError("source_count must be an integer")
        if self.source_count < 1:
            raise ValueError("source_count must be positive")
        if not all(isinstance(t, Tensor) for t in (self.key, self.value, self.log_mass)):
            raise TypeError("key, value and log_mass must be tensors")
        if self.key.ndim != 4 or self.value.shape != self.key.shape or min(self.key.shape) < 1:
            raise ValueError("key/value must have the same nonempty [B,H,M,d] shape")
        b, h, m, _ = self.key.shape
        if self.log_mass.shape != (b, h, 1, m):
            raise ValueError("log_mass must have shape [B,H,1,M]")
        if m > self.source_count:
            raise ValueError("slot count cannot exceed source_count")
        if self.key.dtype not in (torch.float32, torch.float64):
            raise ValueError("this adapter currently supports float32/float64 only")
        for tensor in (self.value, self.log_mass):
            if tensor.dtype != self.key.dtype or tensor.device != self.key.device:
                raise ValueError("cache tensors must share dtype and device")
        if not all(bool(torch.isfinite(t).all()) for t in (self.key, self.value, self.log_mass)):
            raise ValueError("cache tensors must be finite")
        mass = self.log_mass.detach().double().exp()
        if not torch.allclose(mass, mass.round(), rtol=1e-5, atol=1e-5) or bool((mass < 0.99999).any()):
            raise ValueError("log_mass must represent positive integer source counts")
        total = mass.sum(dim=-1)
        if not torch.allclose(total, torch.full_like(total, self.source_count), rtol=1e-5, atol=1e-5):
            raise ValueError("per-head mass must sum to source_count")

    @classmethod
    def full(cls, key: Tensor, value: Tensor) -> WeightedKVEntry:
        if key.ndim != 4:
            raise ValueError("full key must have shape [B,H,N,d]")
        b, h, n, _ = key.shape
        return cls(key, value, key.new_zeros((b, h, 1, n)), n)

    @property
    def num_slots(self) -> int:
        return self.key.shape[-2]

    @property
    def tensor_bytes(self) -> int:
        """Logical tensor bytes, not process RSS or total deployment bytes."""
        return sum(t.numel() * t.element_size() for t in (self.key, self.value, self.log_mass))

    def to(self, device=None, dtype=None) -> WeightedKVEntry:
        return type(self)(self.key.to(device=device, dtype=dtype),
                          self.value.to(device=device, dtype=dtype),
                          self.log_mass.to(device=device, dtype=dtype), self.source_count)

    def clone(self, *, detach: bool = False) -> WeightedKVEntry:
        def copy(t):
            return t.detach().clone() if detach else t.clone()
        return type(self)(copy(self.key), copy(self.value), copy(self.log_mass), self.source_count)

    def slice_batch(self, index) -> WeightedKVEntry:
        if isinstance(index, int):
            if not -self.key.shape[0] <= index < self.key.shape[0]:
                raise IndexError(index)
            index = [index]
        return type(self)(self.key[index], self.value[index], self.log_mass[index], self.source_count)

    def __getitem__(self, index) -> WeightedKVEntry:
        return self.slice_batch(index)

    @classmethod
    def concat(cls, entries: Sequence[WeightedKVEntry], dim: int = 0) -> WeightedKVEntry:
        if not entries or dim != 0:
            raise ValueError("concat requires entries and operates only on the batch dimension")
        ref = entries[0]
        for entry in entries:
            if (entry.source_count != ref.source_count or entry.key.shape[1:] != ref.key.shape[1:]
                    or entry.key.device != ref.key.device or entry.key.dtype != ref.key.dtype):
                raise ValueError("concatenated caches must share head layout, N, dtype and device")
        return cls(torch.cat([e.key for e in entries]), torch.cat([e.value for e in entries]),
                   torch.cat([e.log_mass for e in entries]), ref.source_count)

    def state_dict(self) -> dict:
        """Plain tensors/primitives for torch.load(weights_only=True)."""
        # CPU slices can still reference a much larger backing storage. Clone
        # before export so saving one batch cannot retain the omitted batches.
        return {"key": self.key.detach().cpu().clone(), "value": self.value.detach().cpu().clone(),
                "log_mass": self.log_mass.detach().cpu().clone(), "source_count": self.source_count}

    @classmethod
    def from_state_dict(cls, state: Mapping) -> WeightedKVEntry:
        if set(state) != {"key", "value", "log_mass", "source_count"}:
            raise ValueError("weighted cache state must contain key/value/log_mass/source_count")
        return cls(**state)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "tracelight-weighted-kv", "version": ARTIFACT_VERSION,
                    "entry": self.state_dict()}, target)

    @classmethod
    def load(cls, path: str | Path) -> WeightedKVEntry:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "tracelight-weighted-kv" or state.get("version") != ARTIFACT_VERSION:
            raise ValueError("unsupported weighted cache artifact")
        return cls.from_state_dict(state["entry"])
