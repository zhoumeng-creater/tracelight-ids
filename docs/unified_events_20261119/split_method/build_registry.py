"""Read existing evidence and write explicit source/chain/member roles. No scores."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPO = WORK.parent / "代码" / "tracelight-ids"
SOURCE = REPO / ".worktrees" / "source-method-route-20261118"
inputs = {}


def read(path):
    raw = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8-sig"))


def main():
    groups = []
    members = []
    for evidence_path in [WORK/'学习任务重设_20261115/来源与事件分配.md',
        WORK/'研究路线全面审查_20261115/总审查与路线决定.md',
        WORK/'独立来源与算法主线_20261118/algorithm_sources.md',
        SOURCE/'docs/SOURCE_METHOD_ROUTE_20261118.md',
        SOURCE/'docs/MODEL_TRAINING_REDESIGN_20261117.md']:
        inputs[str(evidence_path)] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()

    def group(gid, source, implementation, exposure, role, future_role, blockers, **extra):
        groups.append(dict(group_id=gid, source_group=source,
            implementation_group=implementation, exposure=exposure, current_role=role,
            proposed_next_role=future_role, confirmation_ready=False,
            admission_blockers=blockers, **extra))

    def member(mid, gid, unit_type, label, evidence, **extra):
        members.append(dict(member_id=mid, group_id=gid, unit_type=unit_type,
                            target_label=label, evidence=evidence, **extra))

    group("ait:russellmitchell:complete_run", "AIT-LDS", "kyoushi_wpdiscuz_webshell_wphashcrack",
          "actions_logs_features_and_pcap_opened", "opened_development",
          "development_control_and_normal_diagnostics_no_complete_download_training",
          ["Four intranet packet candidates (3544,3574,3575,3578) have invalid captured IP lengths and are barred from complete raw8",
           "Two attack download actions cannot supply complete-flow training targets from these captures",
           "Action-to-flow multiplicity remains; new F/V/C identities not registered"],
          completed_evidence="Unified links and group roles resolved; canonical Zeek -C justified by packet checksums and author interface facts; 38 cloud and 32 intranet control C records qualified",
          interval_utc=["2022-01-21T00:00:00Z", "2022-01-25T00:00:00Z"],
          group_rule="All control, tool downloads, normal ownCloud and website actions in this run remain together for attack-chain evaluation; normal/calibration roles need separate explicit registration.")
    for name in ["harrison", "fox", "santos", "shaw", "wardbeck", "wheeler", "wilson"]:
        group(f"ait:{name}:complete_run", "AIT-LDS", "kyoushi_shared_attack_generator",
              "publisher_metadata_only_in_reviewed_evidence", "reserved_not_admitted" if name == "harrison" else "not_selected_not_admitted",
              "same_generator_cross_run_confirmation_candidate" if name == "harrison" else "no_current_assignment",
              ["exact runtime intervals and attack action success unverified", "comparable normal business and network mapping unverified"],
              independence_scope="new execution/environment only; not independent source or attack implementation")

    normal_path = WORK / "独立来源与算法主线_20261118" / "normal_event_manifest.json"
    normal = read(normal_path)
    for event in normal["events"]:
        member(event["event_id"], "ait:russellmitchell:complete_run", "business_action_candidate",
               "normal_file_download" if event["status"] == "action_server_proxy_confirmed" else "unknown",
               {"path": str(normal_path), "status": event["status"]},
               flow_label=event.get("flow_label"), same_pcap_link_pending=True)
    ait_links_path = WORK / '统一事件与方法落实_20261119/ait/normal_event_connections.json'
    if ait_links_path.exists():
        ait_links = read(ait_links_path)
        by_id = {r['event_id']:r for r in ait_links['events']}
        for row in members:
            if row['member_id'] in by_id:
                linked = by_id[row['member_id']]
                row.update(same_pcap_link_pending=False,
                    association_evidence=str(ait_links_path),
                    packet_candidate_class=linked['packet_candidate_class'],
                    request_second_payload_candidate_class=linked['request_second_proxy_payload_candidate_class'],
                    request_second_payload_zeek_ids_by_mode=linked.get('request_second_payload_zeek_ids_by_mode'),
                    flow_label=None,
                    association_is_pure_file_label=False)

    audit_path = WORK / "独立事件与风险迁移_20261115" / "AIT实件准入核验.json"
    audit = read(audit_path)
    for event in audit["apache"]["commands"]:
        member(f"ait:russellmitchell:webshell:{event['member']}:{event['line']}",
               "ait:russellmitchell:complete_run", "http_request", "author_webshell_command_request",
               {"path": str(audit_path), **event}, execution_confirmed_per_request=False,
               independent_attack_count=0)
    for employee in audit["controlled_normal_business"]:
        for event in employee["actions"]:
            member(f"ait:russellmitchell:normal_page:{event['transition_id']}",
                   "ait:russellmitchell:complete_run", "business_action", "normal_website_open",
                   {"path": str(audit_path), "employee": employee["role"], **event},
                   flow_label=None)
    for name, label, line in [("wphashcrack", "malicious_tool_download_with_unarchive", 400),
                               ("password_list", "attack_dictionary_transfer_action", 410)]:
        member(f"ait:russellmitchell:attack_download:{name}", "ait:russellmitchell:complete_run",
               "actor_action", label, {"path": str(audit_path), "actor_member": audit["actor_evidence"]["member"], "start_line": line},
               flow_label=None, relationship_causality_label=None,
               complete_raw8_training_eligible=False,
               reason="Intranet download candidates 3574/3575/3578 have invalid captured IP lengths; -C does not restore missing bytes")

    group("optc:empire_custom:20190924_continues_20190925", "OpTC", "PowerShell_Empire",
          "full_selected_network_host_hour_and_ground_truth_opened", "opened_development_holdout",
          "development_external_holdout_no_fit_no_threshold_selection",
          ["Normal denominator is the declared 41630-connection business subset, not whole-network traffic; new F/V/C registration remains separate"],
          completed_evidence="42 B1 HTTP and 41588 B2 TLS normal connections and 26 declared business evaluation units assembled",
          group_rule="All agents, lateral descendants, downloads and 09/25 persistence from the 09/24 campaign remain one group, including records outside the currently downloaded hour.")
    op = WORK / "独立来源与算法主线_20261118" / "evidence" / "optc"
    control_path = op / "optc_control_channel_slice.json"
    control = read(control_path)
    for uid in control["confirmed_control_uids"]:
        member(f"optc:20190924:control:{uid}", "optc:empire_custom:20190924_continues_20190925",
               "network_session", "known_control_transport", {"path": str(control_path), "uid": uid},
               executed_command_label=None, independent_attack_count=0)
    download_path = op / "optc_event_slice.json"
    download = read(download_path)
    for idx, row in enumerate(download["candidates"]["http_filename_all"]):
        member(f"optc:20190924:download:{row['uid']}", "optc:empire_custom:20190924_continues_20190925",
               "network_session", "redteam_file_download" if idx == 0 else "repeat_same_file_download_in_redteam_episode",
               {"path": str(download_path), "uid": row["uid"], "fuid": row["resp_fuids"], "ts": row["ts"]},
               documented_action_id=download["action"]["event_id"] if idx == 0 else None,
               source_action_count=1 if idx == 0 else 0, relationship_causality_label=None)
    group("optc:20190919:selected_normal_candidates", "OpTC", "normal_business_unresolved",
          "full_selected_http_files_hour_opened", "qualification_unknown_not_normal_pool",
          "only_explicitly_confirmed_normal_rows_from_11_19_assembly",
          ["author benign partition is not sufficient for six suspicious PE objects"])
    benign_path = op / "benign_file_business.json"
    benign = read(benign_path)
    for row in benign["candidate_file_rows"]:
        member(f"optc:20190919:unknown_pe:{row['fuid']}", "optc:20190919:selected_normal_candidates",
               "file_object", "unknown", {"path": str(benign_path), "fuid": row["fuid"], "uid": row["conn_uids"]},
               usable_as_normal=False)
    group("optc:empire_plain:20190923_complete", "OpTC", "PowerShell_Empire",
          "official_action_pdf_opened_network_not_opened_in_reviewed_evidence", "reserve_not_selected",
          "same_source_same_implementation_episode_candidate",
          ["no assembled event-network pairs", "no verified comparable normal denominator"],
          independence_scope="different episode, not unseen source or unseen implementation")
    normal_gid = 'optc:20190919:declared_normal_business_subset'
    normal_flows = WORK / '统一事件与方法落实_20261119/optc/normal_flow_events.jsonl.gz'
    normal_objects = WORK / '统一事件与方法落实_20261119/optc/normal_http_file_objects.jsonl.gz'
    normal_members = []
    if normal_flows.exists() and normal_objects.exists():
        for path,kind in [(normal_flows,'network_session'),(normal_objects,'http_file_object')]:
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path,'rt',encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    normal_members.append(dict(member_id=row['event_id'],group_id=normal_gid,unit_type=kind,
                        role='opened_development_external_business_holdout_no_fit_no_threshold_selection',
                        source_manifest=str(path),uid=row.get('uid'),fuid=row.get('fuid'),
                        business_episode_group=row.get('business_episode_group'),evidence_grade=row.get('evidence_grade')))
        group(normal_gid,'OpTC','normal_browser_workload',
            'qualification_labels_and_size_overlap_opened_no_model_score',
            'opened_development_external_business_holdout',
            'same_role_no_fit_no_threshold_selection',
            ['selected business subset, not full-network normal denominator'],
            membership_file=str(OUT/'normal_member_registry.jsonl.gz'),
            member_counts=dict(Counter(x['unit_type'] for x in normal_members)),
            interval_utc=['2019-09-19T17:00:00Z','2019-09-19T18:00:00Z'])
    group("optc:meterpreter:20190925_complete_branch", "OpTC", "Meterpreter_reverse_tcp",
          "official_action_pdf_and_selected_raw_qualification_only_no_model_use", "reserved_confirmation_candidate_under_assembly",
          "same_source_implementation_holdout_no_fit",
          ["update.exe actions not yet joined to network and host records", "same-service normal update/control denominator unverified"],
          hosts=["Sysclient0051", "Sysclient0351"],
          metadata_actions_local=["2019-09-25 10:29:42", "2019-09-25 11:23:31", "2019-09-25 14:24:03"],
          group_rule="Both victim hosts, migrations and repeated payload runs are one implementation branch. Exclude 09/24 Empire continuation from this group.",
          independence_scope="different implementation from opened OpTC Empire, same source/testbed; does not establish independence from any unverified legacy reverse-shell implementation")
    meterpreter_path = OUT/'meterpreter_qualification.json'
    if meterpreter_path.exists():
        qualification = read(meterpreter_path)
        groups[-1]['qualification_summary'] = qualification['summary']
        groups[-1]['qualified_action_episode_count'] = len(qualification['summary'].get('hosts_with_complete_download_and_exact_control',[]))
        groups[-1]['event_qualification_ready'] = bool(groups[-1]['qualified_action_episode_count'])
        groups[-1]['admission_blockers'] = [
            'One Sysclient0351 upgrade episode has complete download and exact control; Sysclient0051 control UID remains unjoined in fetched rotations',
            'Normal calibration identities and matched declared confirmation denominator not yet assigned; not a ready end-to-end model comparison',
            'Later branch migrations/reruns remain reserved but are not fully assembled']
        groups[-1]['qualification_path'] = str(meterpreter_path)
        groups[-1]['allowed_qualification_read'] = 'Identity, ground-truth association and completeness only; no algorithm/window/threshold selection'
        groups[-1]['partial_slice_does_not_split_group'] = True
        for row in qualification['download_events']:
            member(row['event_id'],groups[-1]['group_id'],'network_session',
                'redteam_malicious_upgrade_download' if row.get('matches_author_host_payload') else 'unassigned_same_branch_payload_candidate',
                {'path':str(meterpreter_path),'grade':row['evidence_grade']},
                documented_action_id=row.get('action_id'),
                role='reserved_external_historical_confirmation_no_fit_or_selection')
        for row in qualification['matched_control_sessions']:
            member(row['event_id'],groups[-1]['group_id'],'network_session','known_meterpreter_transport',
                {'path':str(meterpreter_path)},role='reserved_external_historical_confirmation_no_fit_or_selection')

    group("dedale:green:normal_prefix_D01_D14", "DEDALE", "author_normal_workload",
          "R25_all_days_scored", "legacy_opened_development_background",
          "normal_fit_or_calibration_pool_only_after_separate_nonoverlapping_allocation", ["no new allocation to fit versus calibration in this registry"])
    group("dedale:green:APT_D15_D28_complete", "DEDALE", "DEDALE_reverse_shell_and_stealth_tool_family_not_reverified",
          "R25_all_days_scored_and_D15_host_network_examined", "legacy_opened_development",
          "preserve_whole_chain_development_holdout_no_new_fit_allocation", ["exact supervised scope must retain author_rule versus host_confirmed evidence levels"],
          group_rule="All D15-D28 attack steps and any other sensor observing them share this group; never train D15 and claim D17 independent confirmation.")
    dedale_path = SOURCE / "configs" / "behavior-transfer-20261114" / "dedale.json"
    dedale = read(dedale_path)
    for day in dedale["sources"]:
        gid = "dedale:green:normal_prefix_D01_D14" if day["day"] <= 14 else "dedale:green:APT_D15_D28_complete"
        member(day["source_id"], gid, "native_day_file", "mixed_author_labels" if day["day"] >= 15 else "author_normal_prefix",
               {"path": str(dedale_path), "date": day["date"], "historical_registration_role": day["role"]},
               current_exposure="R25_scored", stale_unscored_field_overridden=True)
    check_path = REPO / ".worktrees" / "readiness-replay-r25" / "docs" / "behavior-transfer" / "dedale-D15-source-check.json"
    check = read(check_path)
    for uid, grade in check["event_evidence"]["uid_evidence"].items():
        member(f"dedale:D15:{uid}", "dedale:green:APT_D15_D28_complete", "network_session",
               "download" if uid in ["CI7cJo10RfMA3xxRC5", "CnOKRT2GLSyyTe1kRf"] else "control_transport",
               {"path": str(check_path), "uid": uid, "grade": grade})

    families = {"A": "Sliver_TLS", "B": "Merlin_QUIC", "C": "Mythic_HTTP", "D": "Merlin_HTTP2_TLS"}
    for profile, impl in families.items():
        group(f"tqh:{profile}:12_complete_units", "TQH-C2", "Merlin" if profile in "BD" else impl.split('_')[0],
              "R23_all_48_scored; A_B_D_used_in_later_development", "legacy_opened_development_only",
              "preserve_original_registered_F_V_C_E_roles_no_new_confirmation", [],
              profile=impl, units_are_independent_implementations=False)
    tqh_path = WORK / "学习任务重设_20261115" / "TQH完整单元时间分配核验.json"
    tqh = read(tqh_path)
    for unit in tqh["units"]:
        profile = unit["profile"]
        name = unit["unit"]
        if profile in "AB" and "_i30_" in name:
            role = "R30_F_fit"
        elif profile in "AB" and "_i300_" in name:
            role = "R30_V_selection"
        elif profile == "B" and "_i1800_" in name:
            role = "R30_C_normal_only"
        elif profile == "B" and "_i3600_" in name:
            role = "R30_E_B_opened_development_evaluation"
        elif profile == "D":
            role = "R30_E_D_opened_development_evaluation"
        else:
            role = "R23_opened_not_used_in_R30_FVCE"
        member(f"tqh:{name}", f"tqh:{profile}:12_complete_units", "complete_capture_unit", "author_c2_not_per_command_or_download",
               {"path": str(tqh_path), "start_utc": unit["start_utc"], "end_utc": unit["end_utc"], "label_counts": unit["labels"]},
               preserved_legacy_role=role, reassignment_for_new_fit=False)

    group("unraveled:APT_complete_trajectory", "Unraveled_DAPT2021", "Covenant_Sliver_Metasploit_trajectory_not_individually_attributed",
          "2021_06_22_net1013x_labeled_flow_prefix_opened", "qualification_opened_candidate",
          "audited_cross_source_holdout_no_fit_after_label_provenance", ["exact per-flow label derivation and attack execution records not verified", "normal management and download execution must be matched"],
          independence_scope="no-fit external source possible, not a pristine unseen APT chain because a sample already informed design",
          group_rule="Entire APT trajectory and duplicate gateway/subnet observations grouped; no train/test split by week or sensor")
    for name in ["Amateur", "Skilled_Hackers"]:
        group(f"unraveled:{name}:complete_trajectory", "Unraveled_DAPT2021", "not_verified",
              "metadata_only", "not_selected_not_admitted", "no_current_assignment",
              ["not verified as a sustained control plus completed download target"])
    group("cesnet:CC25_complete_source", "CESNET-CC25", "botnet_families_six_reported",
          "botnet_flow_archive_and_normal_schema_opened", "periodicity_reference_development",
          "periodic_control_method_reference_only", ["not per-download/action truth", "benign and botnet captured at different locations"])
    group("ieu:C2D_released_sequences", "IEU-C2D", "multiple_tools_metadata",
          "public_sequence_schema_opened", "method_neighbour_only", "no_current_event_split",
          ["normal business and original action-network provenance not supplied in reviewed release"])
    group("iot23:legacy_opened_captures", "IoT-23", "mixed_known_families",
          "legacy_training_or_scored", "legacy_reference_only", "no_new_confirmation",
          ["download labels partly rely on byte/endpoint rules"],
          capture_ids=["1-1", "8-1", "4-1", "5-1", "34-1", "35-1", "42-1", "43-1", "44-1", "48-1", "49-1", "52-1", "60-1"])
    group("iot23:remaining_nine", "IoT-23", "mixed_unopened_candidates",
          "publisher_metadata_known_no_local_score_record_in_prior_audit", "not_selected",
          "not_a_control_plus_download_confirmation_pool", ["no FileDownload targets in these nine captures"],
          capture_ids=["3-1", "7-1", "9-1", "17-1", "20-1", "21-1", "33-1", "36-1", "39-1"])
    group('iotsfc:106:all14_dateblocks','IoT-SFC','author_scan_and_mixed_attack_workload',
        'R25_all_25_backups_deduplicated_14_blocks_scored','legacy_opened_development_reference',
        'no_new_confirmation',['no pure control-plus-download event target'],
        group_rule='Other sensors of the same author event are not independent confirmation.')

    ids = [r["member_id"] for r in members]
    assert len(ids) == len(set(ids))
    gids = {r["group_id"] for r in groups}
    assert all(r["group_id"] in gids for r in members)
    assert len(control["confirmed_control_uids"]) == 24
    assert sum(r["target_label"] == "normal_file_download" for r in members) == 31
    assert len(tqh["units"]) == 48
    assert len({r['member_id'] for r in normal_members}) == len(normal_members)
    assert not (set(ids) & {r['member_id'] for r in normal_members})
    unified_path = WORK/'统一事件与方法落实_20261119/unified/summary.json'
    unified_snapshot = read(unified_path) if unified_path.exists() else None
    for path in [WORK/'统一事件与方法落实_20261119/ait/checksum_semantics.json',
                 WORK/'统一事件与方法落实_20261119/ait_attack/checksum_semantics.json']:
        if path.exists():
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
        evidence_commit="74b5425", status="concrete_source_chain_registry_not_training_registration",
        model_calls=0, source_chain_roles=groups, member_registry=members,
        normal_member_sidecar=str(OUT/'normal_member_registry.jsonl.gz'),
        unified_evidence_snapshot=unified_snapshot,
        canonical_new_event_membership=str(WORK/'统一事件与方法落实_20261119/unified/split_assignments.jsonl.gz'),
        counts=dict(groups=len(groups), members=len(members), unit_types=dict(Counter(r["unit_type"] for r in members)),
                    normal_sidecar_members=len(normal_members),
                    event_qualified_reserved_groups=sum(bool(g.get('event_qualification_ready')) for g in groups),
                    confirmation_ready_groups=0),
        mixed_unit_warning="Member rows mix actions, network sessions and capture/day units. Their sum is not a number of independent events.",
        confirmation_readiness_meaning='event_qualified_reserved_groups counts actual joined event candidates; confirmation_ready_groups requires the full declared normal denominator, calibration and final method contract, not merely unopened data.',
        training_and_calibration=dict(new_fit_authorized_by_this_file=False,
            new_calibration_assignment=None,
            reason="Unified event/observation/link membership and declared 41630-flow normal subset are assembled. New fit/validation/calibration identities remain unregistered; AIT download full-flow inputs are specifically blocked."),
        source_sha256=inputs)
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT/'normal_member_registry.jsonl.gz','wt',encoding='utf-8') as f:
        for row in normal_members:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (OUT / "split_registry.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=True))


if __name__ == "__main__":
    main()
