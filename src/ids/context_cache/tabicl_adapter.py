"""Pinned TabICLv2 tensor bridge for complete and QCOC-compressed caches.

Inputs are *already prepared* numeric tables, not raw CSV/sklearn inputs. The
caller owns frozen preprocessing and maps normal/attack to 0/1. No checkpoint
download, data fitting, class permutation or ensemble generation occurs here.
Stage 1/2 use the official implementation. Stage 3 retains its learned layers
but executes attention over our explicit weighted entries, outside upstream's
KVCache container (which would discard mass and source_count on concatenation).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .attention import prepare_query, weighted_attention
from .cache import WeightedKVEntry
from .upstream import require_pinned_tabicl


_FORMAT = "tracelight-tabicl-numeric-cache-v1"


def _official_types():
    # Verifies installed sources; never installs packages or loads weights.
    provenance = require_pinned_tabicl()
    from tabicl._model.tabicl import TabICL
    from tabicl._model.kv_cache import KVCache, KVCacheEntry
    from tabicl._model.inference_config import InferenceConfig
    return TabICL, KVCache, KVCacheEntry, InferenceConfig, provenance


def _configuration(model, supplied=None):
    if supplied is not None:
        config = dict(supplied)
    else:
        config = {}
        for name, parameter in inspect.signature(type(model).__init__).parameters.items():
            if name == "self":
                continue
            if hasattr(model, name):
                config[name] = getattr(model, name)
            elif parameter.default is not inspect.Parameter.empty:
                config[name] = parameter.default
            else:
                raise ValueError(f"provide model_config for constructor parameter {name}")
    if config.get("activation", "gelu") not in ("gelu", "relu"):
        raise ValueError("persisted adapters support string gelu/relu activation only")
    return config


def _device_dtype(model):
    first = next(model.parameters())
    if first.dtype not in (torch.float32, torch.float64):
        raise ValueError("the first cache adapter supports float32/float64 only")
    return first.device, first.dtype


def _inference_config(model):
    from tabicl._model.inference_config import InferenceConfig
    device, _ = _device_dtype(model)
    stage = dict(device=device, use_amp=False, use_fa3=False, offload=False, use_async=False)
    return InferenceConfig(COL_CONFIG=dict(stage), ROW_CONFIG=dict(stage), ICL_CONFIG=dict(stage))


def _validate_x(x, model, *, batch=None, features=None):
    if not isinstance(x, Tensor) or x.ndim != 3 or min(x.shape) < 1:
        raise ValueError("prepared numeric input must be nonempty [B,T,F]")
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError("prepared numeric inputs must contain finite floating-point values")
    if batch is not None and x.shape[0] != batch:
        raise ValueError("query batch must match the context batch; use slice_batch explicitly")
    if features is not None and x.shape[2] != features:
        raise ValueError("query feature count differs from the prepared support contract")
    device, dtype = _device_dtype(model)
    return x.to(device=device, dtype=dtype)


def _project_query(attention, normalized_query, source_count, rope=None):
    width, heads = attention.embed_dim, attention.num_heads
    bias = None if attention.in_proj_bias is None else attention.in_proj_bias[:width]
    q = F.linear(normalized_query, attention.in_proj_weight[:width], bias)
    q = q.reshape(*q.shape[:-1], heads, width // heads).transpose(-3, -2)
    if rope is not None:
        q = rope.rotate_queries_or_keys(q)
    return prepare_query(q, source_count, attention.ssmax_layer)


def _pack_columns(cache):
    # Clone after CPU transfer: a batch slice must not serialize its full storage.
    return {str(i): {"key": e.key.detach().cpu().clone(), "value": e.value.detach().cpu().clone()}
            for i, e in cache.kv.items()}


class CompactPredictor:
    """Single-view binary predictor with official columns/rows and weighted ICL.

    B can contain multiple separately prepared contexts with equal N/F. It is
    not a sklearn ensemble: there are no class or feature permutations. This
    object shares frozen backbone weights with the FullCacheAdapter that made
    it; serialized files contain their own weights and no source rows or phi Q.
    """

    def __init__(self, model, *, model_config, column_cache, entries, train_shape,
                 provenance=None, cache_kind="compiled"):
        _, _, _, _, source = _official_types()
        if int(model.max_classes) < 2:
            raise ValueError("a classification backbone with at least two output classes is required")
        self.model = model.eval().requires_grad_(False)
        self.model_config = _configuration(model, model_config)
        self.column_cache = column_cache
        self.entries = dict(entries)
        self.train_shape = tuple(int(v) for v in train_shape)
        self.provenance = dict(provenance or {})
        self.upstream = source
        self.cache_kind = cache_kind
        if len(self.train_shape) != 3 or min(self.train_shape) < 1:
            raise ValueError("invalid support train_shape")
        blocks = self.model.icl_predictor.tf_icl.blocks
        if set(self.entries) != set(range(len(blocks))):
            raise ValueError("one weighted cache entry is required for every ICL layer")
        for i, entry in self.entries.items():
            attn = blocks[i].attn
            if (entry.key.shape[0] != self.train_shape[0]
                    or entry.source_count != self.train_shape[1]
                    or entry.key.shape[1] != attn.num_heads
                    or entry.key.shape[-1] != attn.embed_dim // attn.num_heads):
                raise ValueError("weighted cache layout differs from the model/support contract")
        _device_dtype(self.model)

    @torch.no_grad()
    def encode_queries(self, query_x):
        """Run the unchanged official first two stages without query labels."""
        x = _validate_x(query_x, self.model, batch=self.train_shape[0], features=self.train_shape[2])
        config = _inference_config(self.model)
        columns = self.model.col_embedder.forward_with_cache(
            x, col_cache=self.column_cache, y_train=None, use_cache=True,
            store_cache=False, mgr_config=config.COL_CONFIG,
        )
        return self.model.row_interactor(columns, mgr_config=config.ROW_CONFIG)

    def stage3_logits(self, representations, *, entries=None):
        """Differentiable cache query path; backbone parameters remain frozen.

        Keeping this outside the upstream inference manager also permits later
        cache-Jacobian checks. It does not fit or update any parameter.
        """
        if self.model.training:
            raise ValueError("cached prediction requires the frozen model in eval mode")
        cache = self.entries if entries is None else entries
        predictor = self.model.icl_predictor
        encoder = predictor.tf_icl
        out = representations
        for i, block in enumerate(encoder.blocks):
            entry = cache[i]
            query = block.norm1(out) if block.norm_first else out
            phi = _project_query(block.attn, query, entry.source_count, encoder.rope)
            attended = weighted_attention(phi, entry)
            attended = attended.transpose(-3, -2).contiguous().reshape_as(out)
            attended = F.linear(attended, block.attn.out_proj.weight, block.attn.out_proj.bias)
            attended = block.dropout1(attended)
            if block.norm_first:
                out = out + attended
                out = out + block._ff_block(block.norm2(out))
            else:
                out = block.norm1(out + attended)
                out = block.norm2(out + block._ff_block(out))
        if predictor.norm_first:
            out = predictor.ln(out)
        return predictor.decoder(out)[..., :2]

    @torch.no_grad()
    def predict_logits(self, query_x):
        return self.stage3_logits(self.encode_queries(query_x))

    @torch.no_grad()
    def score(self, query_x):
        logits = self.predict_logits(query_x)
        return logits[..., 1] - logits[..., 0]

    def to(self, device, dtype=None):
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise ValueError("the first cache adapter supports float32/float64 only")
        current_device, current_dtype = _device_dtype(self.model)
        target_device = torch.device(device)
        target_dtype = current_dtype if dtype is None else dtype
        if target_device != current_device or target_dtype != current_dtype:
            # Full/compact controls initially share frozen weights. Moving one
            # must not mutate the other's model while its cache stays behind.
            self.model = copy.deepcopy(self.model)
        self.model.to(device=device, dtype=dtype)
        self.column_cache = self.column_cache.to(device=device, dtype=dtype)
        self.entries = {i: e.to(device=device, dtype=dtype) for i, e in self.entries.items()}
        return self

    def slice_batch(self, start, end):
        if not 0 <= start < end <= self.train_shape[0]:
            raise ValueError("batch slice must be nonempty and within the context batch")
        return type(self)(
            self.model, model_config=self.model_config, column_cache=self.column_cache[slice(start, end)],
            entries={i: e.slice_batch(slice(start, end)) for i, e in self.entries.items()},
            train_shape=(end - start, *self.train_shape[1:]), provenance=self.provenance,
            cache_kind=self.cache_kind,
        )

    def size_report(self):
        def size(t):
            return t.numel() * t.element_size()
        weights = sum(size(t) for t in self.model.parameters()) + sum(size(t) for t in self.model.buffers())
        columns = sum(size(e.key) + size(e.value) for e in self.column_cache.kv.values())
        kv = sum(size(e.key) + size(e.value) for e in self.entries.values())
        mass = sum(size(e.log_mass) for e in self.entries.values())
        return {"model_tensor_bytes": weights, "column_cache_tensor_bytes": columns,
                "icl_key_value_tensor_bytes": kv, "icl_mass_tensor_bytes": mass,
                "cache_tensor_bytes": columns + kv + mass, "total_tensor_bytes": weights + columns + kv + mass,
                "original_support_count": self.train_shape[1],
                "slots_per_layer": {str(i): e.num_slots for i, e in self.entries.items()},
                "includes_external_preprocessing": False, "is_process_rss": False}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _, dtype = _device_dtype(self.model)
        payload = {
            "format": _FORMAT, "model_config": self.model_config,
            "model_state": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "column_cache": _pack_columns(self.column_cache),
            "icl_cache": {str(i): e.state_dict() for i, e in self.entries.items()},
            "train_shape": list(self.train_shape), "num_classes": 2,
            "dtype": str(dtype).removeprefix("torch."), "cache_kind": self.cache_kind,
            "input_contract": "prepared-numeric-single-view; normal=0, attack=1; no class permutation",
            "upstream": self.upstream, "provenance": self.provenance,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path, *, device="cpu"):
        TabICL, KVCache, KVCacheEntry, _, source = _official_types()
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != _FORMAT or payload.get("num_classes") != 2:
            raise ValueError("not a supported binary numeric cache artifact")
        if payload.get("upstream") != source:
            raise ValueError("artifact upstream source differs from the verified pinned implementation")
        dtypes = {"float32": torch.float32, "float64": torch.float64}
        if payload.get("dtype") not in dtypes:
            raise ValueError("unsupported artifact dtype")
        model = TabICL(**payload["model_config"]).to(dtype=dtypes[payload["dtype"]])
        model.load_state_dict(payload["model_state"], strict=True)
        columns = KVCache(kv={int(i): KVCacheEntry(**values) for i, values in payload["column_cache"].items()})
        entries = {int(i): WeightedKVEntry.from_state_dict(values) for i, values in payload["icl_cache"].items()}
        result = cls(model, model_config=payload["model_config"], column_cache=columns,
                     entries=entries, train_shape=payload["train_shape"],
                     provenance=payload.get("provenance"), cache_kind=payload["cache_kind"])
        return result.to(device)


class FullCacheAdapter:
    """Build/capture one official full-support pass, then compile independent KV.

    ``support_queries`` is an offline compilation object. It is never saved by
    CompactPredictor.save, including when exporting the full-cache baseline.
    """

    def __init__(self, model, *, model_config=None, provenance=None):
        TabICL, _, _, _, source = _official_types()
        if not isinstance(model, TabICL) or model.max_classes < 2:
            raise ValueError("expected the pinned official TabICL classification model")
        self.model = model.eval().requires_grad_(False)
        self.model_config = _configuration(model, model_config)
        self.provenance = dict(provenance or {})
        self.upstream = source
        self.cache = None
        self.entries = {}
        self.support_queries = {}
        _device_dtype(model)

    @classmethod
    def from_checkpoint(cls, path, *, device="cpu", expected_sha256=None):
        """Read an existing official config/state_dict file; never auto-download."""
        TabICL, _, _, _, _ = _official_types()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"local checkpoint required: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise ValueError("local checkpoint SHA-256 does not match the expected file")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not {"config", "state_dict"} <= checkpoint.keys():
            raise ValueError("official checkpoint must contain config and state_dict")
        model = TabICL(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        return cls(model, model_config=checkpoint["config"], provenance={
            "checkpoint_name": path.name, "checkpoint_sha256": digest,
            "checkpoint_hash_was_required": expected_sha256 is not None,
        })

    @torch.no_grad()
    def build(self, support_x, support_y):
        if _device_dtype(self.model)[1] != torch.float32:
            # Pinned OneHotAndLinear embeds .float() labels. Do not silently
            # modify its official arithmetic to make a double build run.
            raise ValueError("official support-cache construction requires a float32 backbone")
        x = _validate_x(support_x, self.model)
        if not isinstance(support_y, Tensor) or support_y.shape != x.shape[:2]:
            raise ValueError("support labels must have shape [B,N]")
        if not all(torch.equal(torch.unique(row).cpu(), torch.tensor([0, 1], dtype=row.dtype))
                   for row in support_y):
            raise ValueError("every context must contain both prepared labels normal=0 and attack=1")
        y = support_y.to(device=x.device)
        source_count = x.shape[1]
        encoder = self.model.icl_predictor.tf_icl
        captured = {i: [] for i in range(len(encoder.blocks))}
        handles = []

        def capture(index):
            def hook(attention, args, kwargs):
                query = args[0] if args else kwargs["query"]
                if query.ndim != 3 or query.shape[1] != source_count or kwargs.get("cached_kv") is not None:
                    raise ValueError("support capture requires a full uncached support-only pass")
                phi = _project_query(attention, query, source_count, kwargs.get("rope"))
                captured[index].append(phi.detach().cpu().clone())
            return hook

        try:
            for i, block in enumerate(encoder.blocks):
                handles.append(block.attn.register_forward_pre_hook(capture(i), with_kwargs=True))
            self.model.forward_with_cache(
                X_train=x, y_train=y, use_cache=False, store_cache=True, cache_mode="kv",
                inference_config=_inference_config(self.model),
            )
            self.cache = self.model._cache
        finally:
            for handle in handles:
                handle.remove()
            self.model.clear_cache()
        queries = {i: torch.cat(parts, dim=0) for i, parts in captured.items()}
        for phi in queries.values():
            if phi.shape[0] != x.shape[0] or phi.shape[2] != source_count:
                raise RuntimeError("support capture count mismatch (including possible retried batches)")
        self.support_queries = queries
        self.entries = {i: WeightedKVEntry.full(e.key.detach(), e.value.detach())
                        for i, e in self.cache.icl_cache.kv.items()}
        return self

    def _built(self):
        if self.cache is None:
            raise RuntimeError("build the official full support cache first")

    @torch.no_grad()
    def predict_logits(self, query_x):
        self._built()
        x = _validate_x(query_x, self.model, batch=self.cache.train_shape[0], features=self.cache.train_shape[2])
        try:
            return self.model.forward_with_cache(
                X_test=x, cache=self.cache, return_logits=True,
                inference_config=_inference_config(self.model),
            )
        finally:
            self.model.clear_cache()

    def as_predictor(self):
        self._built()
        return CompactPredictor(
            self.model, model_config=self.model_config, column_cache=self.cache.col_cache,
            entries=self.entries, train_shape=self.cache.train_shape,
            provenance=self.provenance, cache_kind="full",
        )

    def compile(self, *, slots, iterations=15, ridge=1e-3, query_chunk_size=256):
        self._built()
        from .compiler import compile_entry
        entries, reports = {}, {}
        for i, entry in self.entries.items():
            phi = self.support_queries[i].to(device=entry.key.device, dtype=entry.key.dtype)
            entries[i], reports[str(i)] = compile_entry(
                entry, phi, slots=slots, iterations=iterations, ridge=ridge,
                query_chunk_size=query_chunk_size,
            )
        predictor = CompactPredictor(
            self.model, model_config=self.model_config, column_cache=self.cache.col_cache,
            entries=entries, train_shape=self.cache.train_shape,
            provenance=self.provenance, cache_kind="qcoc",
        )
        return predictor, {"method": "QCOC-2026-paper-adaptation", "layers": reports,
                           "full": self.as_predictor().size_report(), "compiled": predictor.size_report(),
                           "queries_are_full_support_attention_queries": True}
