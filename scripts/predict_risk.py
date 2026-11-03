"""Replay a trusted V3 student's frozen raw-logit decisions on validation only.

Use only model.joblib artifacts you created or trust: joblib is executable
pickle. Hash verification checks consistency, not trust in an unknown author.
The CPU network retains its trained dtype; its raw outputs are converted to
float64 BEFORE comparison with run.json's frozen raw-logit working point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ids.common import read_json, sha256_file  # noqa: E402
from ids.data import load_partition  # noqa: E402
from ids.models import _check_frame, _network, load_bundle  # noqa: E402
from ids.selection import normalize_feature_schema  # noqa: E402


def verify_risk_research(output):
    from ids.risk_research import verify_risk_research as verify
    return verify(output)


def _no_links(path):
    path = Path(path).absolute()
    for item in [*reversed(path.parents), path]:
        if item.is_symlink():
            raise ValueError(f"Symbolic links are not accepted: {item}")
        try:
            tag = getattr(item.lstat(), "st_reparse_tag", 0)
        except FileNotFoundError:
            continue
        if tag in {0xA0000003, 0xA000000C}:
            raise ValueError(f"Junctions/symbolic links are not accepted: {item}")
    return path.resolve()


def _validation_cohort(frame, identity):
    """Select exactly the saved development cohort and preserve its row order."""
    required = {"sample_id", "feature_hash", "label", "attack_cat"}
    if (not required.issubset(identity) or not identity.columns.is_unique or identity.empty
            or not frame.columns.is_unique or not set(identity).issubset(frame)):
        raise ValueError("Saved validation identity/schema is incomplete")
    for value in (frame["sample_id"], identity["sample_id"]):
        if value.isna().any() or value.duplicated().any() or not value.map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError("Validation sample IDs must be unique nonempty strings")
    if not set(identity["sample_id"]).issubset(set(frame["sample_id"])):
        raise ValueError("Saved validation samples are missing from prepared validation")
    attrs = dict(frame.attrs)
    result = frame.set_index("sample_id", drop=False).loc[identity["sample_id"].tolist()].reset_index(drop=True)
    result.attrs = attrs
    # Compare values independently of nullable/categorical serialization dtype.
    for column in identity:
        left, right = result[column], identity[column].reset_index(drop=True)
        if left.isna().any() or right.isna().any() or not (left.to_numpy() == right.to_numpy()).all():
            raise ValueError("Prepared validation differs from saved identity: " + column)
    return result


def raw_cpu_logits(bundle, frame):
    """Use the saved batch shape/threads and NEVER call predict_scores/sigmoid."""
    import torch
    if bundle.model_name != "mlp" or bundle.network_state is None:
        raise ValueError("V3 inference requires the saved MLP network state")
    chunk, threads = bundle.batch_size, bundle.num_threads
    if type(chunk) is not int or chunk < 1 or type(threads) is not int or threads < 1:
        raise ValueError("Saved inference batch size and thread count must be positive")
    schema = normalize_feature_schema(bundle.feature_schema)
    _check_frame(frame, schema["features"], allow_empty=False)
    matrix = bundle.preprocessor.transform(frame)
    if matrix.shape != (len(frame), bundle.encoded_dim) or not np.isfinite(matrix).all():
        raise ValueError("Preprocessed validation matrix is invalid")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            network = _network(bundle.encoded_dim, bundle.hidden_dims).cpu()
            network.load_state_dict(bundle.network_state)
            network.eval()
            with torch.inference_mode():
                values = np.concatenate([network(torch.from_numpy(matrix[i:i + chunk])).squeeze(1).numpy()
                                         for i in range(0, len(matrix), chunk)]).astype(np.float64)
    finally:
        torch.set_num_threads(previous_threads)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("Network returned invalid raw logits")
    return values


def predict_validation(*, research_dir, condition, seed, data_dir, output, split="validation"):
    if split != "validation":
        raise ValueError("This entry point accepts validation only; reserved partitions are forbidden")
    if not isinstance(condition, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", condition):
        raise ValueError("condition must be a safe declared condition")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    root, data, target = _no_links(research_dir), _no_links(data_dir), _no_links(output)
    if target.exists():
        raise FileExistsError("Prediction output exists; choose a new parquet path")
    if target.suffix.lower() != ".parquet" or target.is_relative_to(root) or target.is_relative_to(data):
        raise ValueError("Output must be a new .parquet outside sealed research and prepared data")
    verify_risk_research(root)
    plan = read_json(root / "plan.json")
    if condition not in plan["config"]["conditions"] or seed not in plan["config"]["seeds"]:
        raise ValueError("Condition/seed is not in the verified research plan")
    manifest_path = _no_links(data / "manifest.json")
    checksum_path = _no_links(data / "manifest.sha256")
    if checksum_path.read_text(encoding="ascii").strip() != sha256_file(manifest_path):
        raise ValueError("Prepared manifest checksum mismatch")
    manifest = read_json(manifest_path)
    if manifest != plan["data"]:
        raise ValueError("Prepared manifest differs from the research plan data binding")
    _no_links(data / "validation.parquet")
    _no_links(data / "audit.json")
    student = _no_links(root / "students" / f"{condition}-s{seed}")
    record = read_json(_no_links(student / "run.json"))
    if record.get("condition") != condition or record.get("seed") != seed:
        raise ValueError("Student identity differs from the selected condition/seed")
    working_point = record.get("working_point", {})
    threshold = working_point.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold)
            or working_point.get("decision_rule") != "float64(raw_logit) >= frozen_raw_logit_threshold"):
        raise ValueError("Missing or incompatible frozen raw-logit threshold contract")
    identity = pd.read_parquet(_no_links(root / "validation-identity.parquet"))
    frame = _validation_cohort(load_partition(data, "validation"), identity)
    bundle = load_bundle(_no_links(student / "model.joblib"), device="cpu")
    raw = raw_cpu_logits(bundle, frame)
    predictions = identity.copy().reset_index(drop=True)
    predictions["cpu_raw_logit"] = raw
    predictions["raw_logit_threshold"] = np.float64(threshold)
    predictions["predicted_label"] = (raw >= np.float64(threshold)).astype(np.int8)
    predictions.attrs = {"scope": "saved_validation_cohort_replay_not_independent_confirmation",
                         "decision_rule": working_point["decision_rule"], "condition": condition, "seed": seed}
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silent replacement even if another process
    # creates the destination after the initial check.
    with target.open("xb") as stream:
        predictions.to_parquet(stream, index=False)
    return {"status": "validation_replay_complete", "condition": condition, "seed": seed,
            "candidate_status": record.get("status"), "data_kind": manifest.get("data_kind"),
            "n_rows": len(predictions), "output": str(target), "sha256": sha256_file(target),
            "threshold": float(threshold), "decision_rule": working_point["decision_rule"],
            "scope": "saved validation cohort only; CPU replay is not independent confirmation",
            "reserved_partitions_read": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = predict_validation(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"V3 validation prediction rejected: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
