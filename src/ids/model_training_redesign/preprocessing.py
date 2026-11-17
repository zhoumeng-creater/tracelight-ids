"""F-only typed inputs with a serialized explicit-protocol intervention."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ids.model_training.preprocessing import CAT_CARDINALITIES, TypedPreprocessor
from ids.model_training.preprocessing import CONTINUOUS as CONTINUOUS


FORMAT = "typed-zeek-c2-redesign-v1"
POLICIES = {"protocol_blind": "constant_unknown", "protocol_aware": "original_mapping"}


@dataclass
class Preprocessor:
    base: TypedPreprocessor
    protocol_policy: str

    @classmethod
    def fit(cls, frame, *, arm, design):
        if arm not in POLICIES:
            raise ValueError(f"Unknown redesign arm: {arm}")
        return cls(TypedPreprocessor.fit(frame, design=design), POLICIES[arm])

    @property
    def cat_cardinalities(self):
        return list(CAT_CARDINALITIES)

    def transform(self, frame):
        numeric, categorical = self.base.transform(frame)
        if self.protocol_policy == "constant_unknown":
            categorical[:, 0] = 5
        elif self.protocol_policy != "original_mapping":
            raise ValueError("Unsupported protocol policy")
        return numeric, np.ascontiguousarray(categorical)

    def to_dict(self):
        return {"format": FORMAT, "protocol_policy": self.protocol_policy,
                "base": self.base.to_dict()}

    @classmethod
    def from_dict(cls, state):
        if state.get("format") != FORMAT or state.get("protocol_policy") not in POLICIES.values():
            raise ValueError("Unsupported redesign preprocessing artifact")
        return cls(TypedPreprocessor.from_dict(state["base"]), state["protocol_policy"])
