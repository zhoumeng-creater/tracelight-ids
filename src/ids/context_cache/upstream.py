"""Verify the official model implementation used by the cache bridge.

Hashes are SHA-256 of LF-normalized files at TABICL_COMMIT. This works for a
pinned source checkout or wheel without assuming a package version equals a
model implementation. There are no installation or network side effects.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from . import TABICL_COMMIT

_MODEL_FILES = {
    "__init__.py": "c24eccee285dd1c5dea721708cd3e52501fe39f29a865fb6cd3d8b448a33ad4f",
    "_cgroup_memory.py": "161bc6813d8e15c657bff0ce169b6b16230fe76e24516737eb341c73953a1495",
    "attention.py": "2406a52571ce33392b003aca185550bc33a8e55797de72bdbf8b4449b09e66f8",
    "embedding.py": "a0c562600fe22944f91b1b71b706198cb9a4c3d9d8aaba67a86f35a85493d322",
    "encoders.py": "a3e0e9bde47e7f587c4a196d124ea631e25755d4f5ac5d6a8568618a4e299a4e",
    "inference.py": "f9e3cf0d83047e06df7d6763bb8d8ee7a00ff1ba6708299a14635831f60ba4fc",
    "inference_config.py": "852abb9cf97e9c82ae42a92b034581f6e9131f3bec70efbd3ce068ac92a77a37",
    "interaction.py": "6f35b5a70cd7b7ea44286faa8921b64c2cc83aca8ed49a4ad8b287abe54cbf27",
    "kv_cache.py": "a7942448631e1770455c826ae159176725f98d2eefc874101c8687f605c57bd8",
    "layers.py": "96cce2e84db33e176d872c7bb7c0df8c305de3eb7a02fcad9f1128f33d319d95",
    "learning.py": "7549dff09c3796a3cb0b55ea03e249342392fa601a681526f7b1efb6a6f1eef2",
    "quantile_dist.py": "7e09ac0c3ce0260dc07cef79b0a98ee63b0785efafbf415fd243fe2981499b99",
    "rope.py": "f59874106743b58330f35699df27ad8d45d4ae692b38c64626f58cdf2da12eab",
    "ssmax.py": "29b3955e2597446cdcfe70f2084a70faa959f0ef510a27ac0668e58943d3d639",
    "tabicl.py": "583786e5ffd70b9eb425bca8e52aed415dadcdf4254c908b3b6fa52026a56491",
}


def require_pinned_tabicl() -> dict:
    try:
        module = importlib.import_module("tabicl._model.tabicl")
    except ImportError as exc:
        raise ImportError(
            "The cache adapter needs the pinned optional TabICLv2 source; "
            "see docs/CONTEXT_CACHE_IMPLEMENTATION_20261106.md. No checkpoint is downloaded."
        ) from exc
    root = Path(module.__file__).resolve().parent
    observed = {}
    for name, expected in _MODEL_FILES.items():
        source = root / name
        actual = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"TabICLv2 source mismatch in {name}; expected commit {TABICL_COMMIT}")
        observed[name] = actual
    return {"repository": "https://github.com/soda-inria/tabicl", "commit": TABICL_COMMIT,
            "verification": "sha256_lf_normalized_model_sources", "model_source_sha256": observed}
