"""Summarize paired operating points without selecting a winning group after evaluation."""

from ids.common import read_json, write_json
from ids.contact_evidence.experiment import GROUPS


def summarize_outcome(root):
    rows = []
    for group in GROUPS:
        result = read_json(root / "evidence/paired" / (group + ".json"))
        calibration = read_json(root / "evidence/probes" / (group + ".json"))["policies"]
        for scope, summary in [("whole_capture", result["capture"]), *result["utc_days"].items()]:
            for arm, stats in summary["comparisons"].items():
                endpoints = stats["endpoints"]
                check = calibration[arm]["within_budget"]
                eligible = bool(check and stats["both_within_budget"])
                pairs = {}
                for name in ("Scanning", "PureCC", "Download", "Heartbeat"):
                    c = endpoints.get(name, {})
                    pairs[name] = {
                        "support": c.get("rows", 0),
                        "missed_by_raw": c.get("rows", 0) - c.get("before", 0),
                        "gained": c.get("gained", 0),
                        "lost": c.get("lost", 0),
                    }
                all_supported = all(pairs[k]["support"] > 0 for k in ("Scanning", "PureCC", "Download"))
                rows.append(
                    {
                        "group": group,
                        "scope": scope,
                        "policy": arm,
                        "early_normal_fp": calibration[arm]["actual_fp"],
                        "early_normal_within_budget": check,
                        "normal_rows": stats["normal_rows"],
                        "allowed_fp": stats["allowed_fp"],
                        "baseline_fp": stats["raw_fp"],
                        "candidate_fp": stats["candidate_fp"],
                        "paired_budget_eligible": eligible,
                        "paired": pairs,
                        "three_targets_present": all_supported,
                        "strict_gain_all_three": (
                            eligible
                            and all(
                                pairs[k]["gained"] > pairs[k]["lost"]
                                for k in ("Scanning", "PureCC", "Download")
                            )
                        )
                        if all_supported
                        else None,
                        "lost_any_native_attack_alert": any(
                            c["lost"] for k, c in endpoints.items() if k not in ("Benign", "Unresolved")
                        ),
                    }
                )
    out = root / "evidence/outcome.json"
    write_json(out, {"capture": "CTU-IoT-Malware-Capture-43-1", "operating_points": rows})
    return [out]
