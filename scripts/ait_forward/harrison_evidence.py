"""Qualify Harrison source logs without opening models or using scores."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import math
import re
import tempfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import yaml

from .qualification import ARCHIVES, RangeReader, official_metadata

ARCHIVE = "harrison_no-pcaps.zip"
MAX_SELECTED_COMPRESSED = 12 * 1024 * 1024
MAX_SELECTED_BYTES = 300 * 1024 * 1024
MAX_TOTAL_RANGES = 24 * 1024 * 1024
BLOCK_BYTES = 1024 * 1024
ACTOR_MEMBER = re.compile(r"gather/(?:internal_employee|remote_employee|ext_user)_\d+/logs/sm\.log")
ACCESS_MEMBER = re.compile(
    r"gather/(?:intranet_server|cloud_share)/logs/apache2/[^/]*access\.log(?:\.\d+)?"
)
APACHE_ROW = re.compile(
    r'(?:(?:\S+:\d+) )?(?P<peer>\S+) - \S+ \[(?P<timestamp>[^\]]+)\] '
    r'"(?P<request>[^"]*)" (?P<status>\d{3})(?: |$)'
)
ACTION_TYPES = {
    "Opened wordpress post": ("Opening wordpress post", "normal_intranet_page_retrieval"),
    "Downloaded file": ("Starting menu action", "normal_owncloud_file_download"),
}


def select_members(archive):
    entries = [entry for entry in archive.infolist() if not entry.is_dir() and entry.file_size]
    access = [entry for entry in entries if ACCESS_MEMBER.fullmatch(entry.filename)]
    names = {
        "dataset.yaml", "processing/config/servers.yaml",
        *(entry.filename for entry in access),
        *(entry.filename.replace("gather/", "labels/", 1) for entry in access),
    }
    picked = [entry for entry in entries if entry.filename in names or ACTOR_MEMBER.fullmatch(entry.filename)]
    if len({entry.filename for entry in picked}) != len(picked):
        raise ValueError("Selected archive has duplicate exact member names")
    if sum(entry.compress_size for entry in picked) > MAX_SELECTED_COMPRESSED:
        raise ValueError("Selected source members exceed the compressed evidence budget")
    if sum(entry.file_size for entry in picked) > MAX_SELECTED_BYTES:
        raise ValueError("Selected source members exceed the decoded evidence budget")
    if not {"dataset.yaml", "processing/config/servers.yaml"}.issubset(
        {entry.filename for entry in picked}
    ):
        raise ValueError("Missing dataset or service clock manifest")
    return sorted(picked, key=lambda entry: entry.filename)


def consume(source, *, destination=None):
    digest = hashlib.sha256()
    actual_crc = 0
    actual_size = 0
    while block := source.read(BLOCK_BYTES):
        if destination is not None:
            destination.write(block)
        digest.update(block)
        actual_crc = zlib.crc32(block, actual_crc)
        actual_size += len(block)
    return actual_size, actual_crc, digest.hexdigest()


def verify_member(result, entry):
    size, crc, _ = result
    if size != entry.file_size or crc != entry.CRC:
        raise ValueError(f"ZIP member length or CRC differs: {entry.filename}")


def stage_member(archive, entry, raw_dir):
    destination = (raw_dir / entry.filename).resolve()
    if not destination.is_relative_to(raw_dir):
        raise ValueError("Archive member resolves outside the evidence directory")
    if destination.exists():
        with destination.open("rb") as previous:
            result = consume(previous)
        verify_member(result, entry)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".member-", dir=destination.parent) as scratch:
            partial = Path(scratch) / "payload"
            with archive.open(entry) as incoming, partial.open("xb") as outgoing:
                result = consume(incoming, destination=outgoing)
            verify_member(result, entry)
            partial.replace(destination)
    return {
        "path": entry.filename, "bytes": result[0], "zip_crc32": f"{result[1]:08x}",
        "sha256": result[2], "compressed_bytes": entry.compress_size,
    }


def stage_sources(raw_dir):
    if raw_dir.exists() and not raw_dir.is_dir():
        raise ValueError("Raw evidence destination must be a directory")
    size, published_checksum = ARCHIVES[ARCHIVE]
    report = {
        "schema": "ait-harrison-source-manifest-v1",
        "official_metadata": official_metadata(),
        "archive": ARCHIVE, "archive_bytes": size, "published_checksum": published_checksum,
        "scope": "normal_actor_service_clock_and_author_web_labels",
        "raw_root": str(raw_dir),
    }
    with RangeReader(ARCHIVE, size, total_byte_limit=MAX_TOTAL_RANGES) as reader:
        with zipfile.ZipFile(reader) as archive:
            picked = select_members(archive)
            report["members"] = [stage_member(archive, entry, raw_dir) for entry in picked]
        report["range_receipts"] = reader.receipts
    return report


def write_new_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def json_lines(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            yield line_number, json.loads(line)


def checked_source_paths(manifest):
    if manifest["schema"] != "ait-harrison-source-manifest-v1" or manifest["archive"] != ARCHIVE:
        raise ValueError("Unexpected Harrison source manifest")
    paths = {}
    root = Path(manifest["raw_root"]).resolve()
    for member in manifest["members"]:
        name = member["path"]
        path = (root / name).resolve()
        if name in paths or not path.is_relative_to(root):
            raise ValueError("Invalid Harrison source path")
        with path.open("rb") as source:
            size, crc, sha256 = consume(source)
        if (size, f"{crc:08x}", sha256) != (member["bytes"], member["zip_crc32"], member["sha256"]):
            raise ValueError(f"Staged Harrison source changed: {name}")
        paths[name] = path
    return paths


def get_clock(paths, roster):
    dataset = yaml.safe_load(paths["dataset.yaml"].read_text(encoding="utf-8"))
    servers = yaml.safe_load(paths["processing/config/servers.yaml"].read_text(encoding="utf-8"))
    roles = {"intranet_server", "cloud_share", *(member.split("/")[1] for member in roster)}
    zones = {role: servers[role].get("timezone") for role in roles}
    if set(zones.values()) != {"UTC"}:
        raise ValueError("Dataset's naive bounds lack a consistent UTC service/actor declaration")
    start, end = (datetime.fromisoformat(str(dataset[key])) for key in ("start", "end"))
    start, end = (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
                  for value in (start, end))
    if start >= end:
        raise ValueError("Dataset clock bounds are invalid")
    return servers, start.timestamp(), end.timestamp()


def object_key(path, query):
    if path.startswith("/remote.php/webdav/"):
        obj = unquote(path.removeprefix("/remote.php/webdav"))
        parent, _, name = obj.rpartition("/")
        return ("file", parent.rstrip("/"), name)
    if path == "/index.php/apps/files/ajax/download.php":
        values = parse_qs(query)
        if len(values.get("files", [])) == 1 and len(values.get("dir", [])) == 1:
            return ("dir", values["dir"][0].rstrip("/"), values["files"][0])
    return None


def collect_actions(paths, roster, servers, start, end):
    actions = []
    per_actor = {}
    for member in roster:
        role = member.split("/")[1]
        actor_ip = servers[role].get("default_ipv4_address")
        transitions = collections.defaultdict(list)
        records = collections.Counter()
        for line, row in json_lines(paths[member]):
            records["rows"] += 1
            ts = row.get("timestamp")
            message = row.get("message")
            transition = row.get("transition_id")
            if not isinstance(ts, (float, int)) or not start <= ts < end:
                continue
            records["in_interval"] += 1
            if not transition:
                continue
            if message in {"Opening wordpress post", "Starting menu action"}:
                transitions[(transition, message)].append((line, ts))
                continue
            if message not in ACTION_TYPES:
                continue
            intent, kind = ACTION_TYPES[message]
            records[f"{kind}_completed"] += 1
            begins = [item for item in transitions[(transition, intent)] if item[1] <= ts]
            if len(begins) != 1:
                records[f"{kind}_without_unique_intent"] += 1
                continue
            begin_line, begin_time = begins[0]
            if kind == "normal_intranet_page_retrieval":
                url = urlsplit(row["wordpress_post"]["link"])
                key = (url.path, url.query, actor_ip)
                service = "intranet_server"
            else:
                obj = row["file"]
                key = (obj["file_type"], obj["path"].rstrip("/"), obj["name"])
                service = "cloud_share"
            actions.append({
                "event_id": f"ait:harrison:normal:{role}:{line}", "kind": kind,
                "service": service, "actor_role": role,
                "actor_member": member, "intent_line": begin_line, "completion_line": line,
                "intent_unix": begin_time, "completion_unix": ts,
                "association_key": key, "service_candidates": [],
            })
        per_actor[role] = dict(records)
    return actions, per_actor


def collect_service_evidence(paths, servers, actions, start, end):
    per_service = {}
    requests = []
    by_key = collections.defaultdict(list)
    for action in actions:
        by_key[(action["service"], action["association_key"])].append(action)
    for member, path in sorted(paths.items()):
        if not ACCESS_MEMBER.fullmatch(member):
            continue
        service = member.split("/")[1]
        server_ip = servers[service]["default_ipv4_address"]
        label_member = member.replace("gather/", "labels/", 1)
        labeled = {}
        if label_member in paths:
            for label_line, row in json_lines(paths[label_member]):
                if row.get("labels"):
                    labeled.setdefault(row["line"], []).append((label_line, row["labels"]))
        observed_annotations = set()
        counts = collections.Counter()
        offsets = collections.Counter()
        first = last = None
        with path.open(encoding="utf-8") as stream:
            for line, record in enumerate(stream, 1):
                counts["lines"] += 1
                match = APACHE_ROW.match(record)
                if not match:
                    counts["unparsed"] += 1
                    continue
                fields = match.groupdict()
                clock = datetime.strptime(fields["timestamp"], "%d/%b/%Y:%H:%M:%S %z")
                ts = clock.timestamp()
                offsets[fields["timestamp"][-5:]] += 1
                first = ts if first is None else min(first, ts)
                last = ts if last is None else max(last, ts)
                annotations = labeled.get(line, [])
                if annotations:
                    observed_annotations.add(line)
                if not start <= ts < end:
                    counts["outside_simulation"] += 1
                    continue
                counts["in_interval"] += 1
                status = int(fields["status"])
                control_label_refs = [
                    label_line for label_line, labels in annotations if "webshell_cmd" in labels
                ]
                if control_label_refs:
                    counts["labeled_control_requests"] += 1
                    if 200 <= status < 300:
                        counts["labeled_control_successful_http_response"] += 1
                    requests.append({
                        "event_id": f"ait:harrison:control_request:{Path(member).name}:{line}",
                        "kind": "author_labeled_control_request",
                        "label_member": label_member, "label_lines": control_label_refs,
                        "request_member": member, "request_line": line, "request_unix": ts,
                        "service": service, "peer_ip": fields["peer"], "server_ip": server_ip,
                        "http_response_status": status,
                        "request_sha256": hashlib.sha256(fields["request"].encode()).hexdigest(),
                    })
                if annotations or status != 200:
                    continue
                parts = fields["request"].split()
                if len(parts) != 3 or parts[0] != "GET":
                    continue
                url = urlsplit(parts[1])
                if service == "intranet_server":
                    key = (url.path, url.query, fields["peer"])
                else:
                    key = object_key(url.path, url.query)
                for action in by_key.get((service, key), ()):
                    if math.floor(action["intent_unix"]) <= ts <= action["completion_unix"]:
                        action["service_candidates"].append({
                            "request_member": member, "request_line": line, "request_unix": ts,
                            "peer_ip": fields["peer"], "server_ip": server_ip,
                            "request_sha256": hashlib.sha256(fields["request"].encode()).hexdigest(),
                        })
        counts["unresolved_nonempty_author_label_line_refs"] = len(labeled.keys() - observed_annotations)
        per_service[member] = {
            "counts": dict(counts), "observed_utc_offsets": dict(offsets),
            "first_request_unix": first, "last_request_unix": last,
            "author_label_source_member": label_member if label_member in paths else None,
        }
    return requests, per_service


def write_jsonl(path, rows):
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with path.open("rb") as stream:
        _, _, sha256 = consume(stream)
    return {"path": str(path), "sha256": sha256, "rows": len(rows)}


def qualify_sources(manifest_path, output):
    with manifest_path.open("rb") as stream:
        raw = stream.read()
    manifest = json.loads(raw)
    paths = checked_source_paths(manifest)
    roster = sorted(member for member in paths if ACTOR_MEMBER.fullmatch(member))
    servers, start, end = get_clock(paths, roster)
    actions, by_actor = collect_actions(paths, roster, servers, start, end)
    control_requests, by_service = collect_service_evidence(paths, servers, actions, start, end)
    multiplicities = collections.Counter(
        (request["request_member"], request["request_line"])
        for action in actions for request in action["service_candidates"]
    )
    counts = collections.Counter()
    for action in actions:
        action.pop("association_key")
        for request in action["service_candidates"]:
            request["candidate_normal_actor_count"] = multiplicities[
                request["request_member"], request["request_line"]
            ]
        counts[action["kind"], len(action["service_candidates"])] += 1
    output.mkdir(parents=True, exist_ok=False)
    proof = {
        "schema": "ait-harrison-source-qualification-v1",
        "admission_status": "not_admitted",
        "source_manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(raw).hexdigest()},
        "time": {
            "simulation_start_unix": start, "simulation_end_exclusive_unix": end,
            "time_evidence": "declared UTC server/actor configurations and explicit Apache offsets",
            "network_export_clock": "unverified",
        },
        "normal": {
            "actor_logs": by_actor, "service_access_logs": by_service,
            "counts_by_service_match_cardinality": [
                {"kind": kind, "candidates": number, "actions": count}
                for (kind, number), count in sorted(counts.items())
            ],
            "qualification": "affirmative actor/service candidates only, no normal flow identities",
        },
        "control": {
            "author_labeled_request_candidates": len(control_requests),
            "successful_http_responses": sum(200 <= row["http_response_status"] < 300
                                             for row in control_requests),
            "qualification": "HTTP request labels, no command execution or network flow claim",
        },
        "download": {
            "qualification": "unavailable",
            "reason": "These source members do not prove a completed malicious transfer; "
                      "an independent completion anchor and network association are required.",
        },
        "artifacts": {
            "normal_action_candidates": write_jsonl(output / "normal_action_candidates.jsonl.gz", actions),
            "control_request_annotations": write_jsonl(
                output / "control_request_annotations.jsonl.gz", control_requests
            ),
        },
    }
    write_new_report(output / "source_qualification.json", proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="Stage the authorized source members here")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--qualify-to", type=Path, help="Read a staged manifest and create a source candidate report")
    args = parser.parse_args()
    if bool(args.raw_dir) == bool(args.qualify_to):
        parser.error("Provide exactly one of --raw-dir or --qualify-to")
    if args.qualify_to:
        report = qualify_sources(args.manifest.resolve(), args.qualify_to.resolve())
        print(f"Qualified {report['control']['author_labeled_request_candidates']} request annotations")
        return
    if args.manifest.exists():
        parser.error("Output manifest already exists")
    report = stage_sources(args.raw_dir.resolve())
    write_new_report(args.manifest, report)
    print(f"Verified {len(report['members'])} source members")


if __name__ == "__main__":
    main()
