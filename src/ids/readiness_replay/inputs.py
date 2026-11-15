"""Bind to the completed R25 run, checking only cached inputs actually consumed."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import time

import pyarrow.parquet as pq

from ids.common import canonical_hash, read_json, sha256_file
from ids.route_batch.protocol import REPO
from . import GROUPS

CONFIG = REPO / "configs/readiness-replay-20261115/inputs.json"


def below(root, relative):
    name = PurePosixPath(relative.replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts or ":" in str(name):
        raise ValueError(f"Expected a relative cache member: {relative}")
    path = (root / str(name)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Cache member leaves the supplied previous root: {relative}")
    return path


def distinct_roots(previous, output):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if previous.is_relative_to(output) or output.is_relative_to(previous):
        raise ValueError("The new output and previous R25 run must be separate directory trees")
    return previous, output


def _receipt(root, stage, binding):
    value = read_json(root / "receipts" / (stage + ".json"))
    if value.get("stage") != stage or value.get("binding") != binding:
        raise ValueError(f"R25 receipt binding changed: {stage}")
    outputs = value.get("outputs", {})
    if not outputs:
        raise ValueError(f"R25 receipt has no outputs: {stage}")
    for name in outputs:
        below(root, name)
    return outputs


def _verify(item):
    path, digest = item
    if not path.is_file():
        return {"path": str(path), "problem": "missing cached input"}
    if sha256_file(path) != digest:
        return {"path": str(path), "problem": "SHA-256 differs from completed R25 receipt"}
    return None


def inspect_previous(previous, *, registration=None, hash_threads=4):
    """Return valid source specs and every unavailable source in one preflight pass.

    Absolute paths embedded in the old source index are not followed. Fixed relative
    paths plus R25 receipt digests allow the entire old output tree to be relocated.
    """
    previous = Path(previous).resolve()
    registration = read_json(CONFIG) if registration is None else registration
    protocol = read_json(previous / "protocol.json")
    binding = canonical_hash(protocol)
    if binding != registration["previous_protocol_sha256"]:
        raise ValueError("Not the registered completed R25 protocol; point --previous to its full output")
    outcome = read_json(previous / "outcome.json")
    if (outcome.get("status") != "complete" or outcome.get("failures")
            or outcome.get("successful_prepared_sources") != len(registration["sources"])
            or outcome.get("evaluated_group_sources") != len(registration["sources"]) * len(GROUPS)):
        raise ValueError("The source run is not the registered complete 42-source, five-learner R25 result")

    started = time.perf_counter()
    prepared, indices = {}, {}
    expected_names = {entry["name"] for entry in registration["sources"]}
    for provider in ("sfc", "dedale"):
        outputs = _receipt(previous, "prepare-" + provider, binding)
        member = "evidence/sources/" + provider + "-index.json"
        problem = _verify((below(previous, member), outputs[member]))
        if problem:
            raise ValueError(problem)
        items = read_json(below(previous, member))
        for spec in items:
            if spec.get("status") != "complete" or spec["name"] in indices:
                raise ValueError("Incomplete or duplicate source in R25 index")
            indices[spec["name"]] = spec
        prepared[provider] = outputs
    if set(indices) != expected_names:
        raise ValueError("R25 source names differ from the frozen 42-source inventory")

    sources, failures, files = [], [], {}
    for registered in registration["sources"]:
        name, provider = registered["name"], registered["provider"]
        old_spec = indices[name]
        try:
            if old_spec["role"] != registered["role"] or old_spec["source_id"] != registered["source_id"]:
                raise ValueError("Source role or identity changed")
            required = {}

            def add(member, outputs):
                if member not in outputs:
                    raise ValueError(f"Required cache member is absent from its R25 receipt: {member}")
                required[member] = outputs[member]

            feature = "local/features/" + name + ".parquet"
            truth = f"local/providers/{provider}/truth/{name}.parquet"
            audit = "evidence/sources/" + name + ".json"
            add(feature, _receipt(previous, "history-" + name, binding))
            for member in (truth, audit):
                add(member, prepared[provider])
            for group in GROUPS:
                add(f"local/scores/{name}/{group}.parquet",
                    _receipt(previous, "score-" + name + "-" + group, binding))
                seal = _receipt(previous, "seal-" + name + "-" + group, binding)
                add(f"local/decisions/{name}/{group}.parquet", seal)
                add(f"evidence/thresholds/{name}/{group}.json", seal)
                judge = _receipt(previous, "judge-" + name + "-" + group, binding)
                for table in ("budget.csv.gz", "paired.csv.gz"):
                    add(f"evidence/evaluation/{name}/{group}/{table}", judge)
            files.update(required)
            sources.append({**registered, "features": feature, "truth": truth, "audit": audit,
                            "input_sha256": required})
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"source": name, "stage": "input_receipts", "error": str(exc)})

    with ThreadPoolExecutor(max_workers=max(1, min(8, hash_threads))) as pool:
        checked = list(pool.map(_verify, [(below(previous, name), digest) for name, digest in files.items()]))
    problems = {str(Path(item["path"]).relative_to(previous).as_posix()): item
                for item in checked if item}
    good = []
    for spec in sources:
        bad = [problems[name] for name in spec["input_sha256"] if name in problems]
        if bad:
            failures.append({"source": spec["name"], "stage": "cached_inputs", "problems": bad,
                             "action": "Use the original lab output with its local features, truth, scores and decisions; a returned summary ZIP is insufficient."})
            continue
        rows = pq.read_metadata(below(previous, spec["features"])).num_rows
        audit = read_json(below(previous, spec["audit"]))
        if rows <= 0 or rows != audit["rows"] or pq.read_metadata(below(previous, spec["truth"])).num_rows != rows:
            failures.append({"source": spec["name"], "stage": "row_counts", "error": "Input rows differ from source audit"})
            continue
        spec["rows"] = rows
        # Numeric features, truth strings, categorical group indices and masks coexist.
        spec["estimated_worker_bytes"] = max(512 * 1024**2, rows * 2200)
        good.append(spec)
    report = {"previous_protocol_sha256": binding, "expected_sources": len(expected_names),
              "usable_sources": len(good), "failures": failures, "files_checked": len(files),
              "input_sha256": files, "seconds": time.perf_counter() - started,
              "fits": 0, "model_forward_calls": 0, "history_builds": 0, "downloads": 0}
    return good, report
