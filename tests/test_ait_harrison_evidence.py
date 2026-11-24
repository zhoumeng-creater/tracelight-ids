import json

from scripts.ait_forward.harrison_download_audit import completion_anchors, http_request_digest
from scripts.ait_forward.harrison_download_join import get_http_digest
from scripts.ait_forward.harrison_evidence import object_key
from scripts.ait_forward.harrison_project import add_observation
from scripts.ait_forward.harrison_review import RAW8, evaluate_connection


def test_transfer_anchor_requires_unique_aligned_intent_and_omits_content():
    messages = (
        ("Downloading item", 11, "one"),
        ("Downloaded item", 12, "one"),
        ("Downloading file", 13, "two"),
        ("Downloading file", 14, "two"),
        ("Downloaded file", 15, "two"),
        ("Downloaded file", 16, "three"),
    )
    entries = [
        (i, json.dumps({
            "message": message, "timestamp": time, "transition_id": transition,
            "private_field": "keep off report",
        }).encode())
        for i, (message, time, transition) in enumerate(messages, 1)
    ]
    anchors, counts = completion_anchors(entries, 10, 20)
    assert counts["paired_completion_anchors"] == 1
    assert counts["completion_without_unique_intent"] == 2
    assert anchors[0]["intent"]["time_unix"] == 11
    assert anchors[0]["completion"]["time_unix"] == 12
    assert "private_field" not in json.dumps(anchors)
    assert "Downloading item" not in json.dumps(anchors)


def test_exact_cloud_object_keys_cover_file_and_folder_forms():
    assert object_key("/remote.php/webdav/Example/test%20file.txt", "") == (
        "file", "/Example", "test file.txt",
    )
    key = object_key(
        "/index.php/apps/files/ajax/download.php", "dir=%2FExample&files=Test%20Folder"
    )
    assert key == ("dir", "/Example", "Test Folder")
    assert object_key(
        "/index.php/apps/files/ajax/download.php", "dir=%2FExample&files=A&files=B"
    ) is None


def test_projection_keeps_every_same_second_candidate_without_labeling():
    anchor = {"request_unix": 100, "candidate_flow_ids": [], "unbounded_candidate_flow_ids": []}
    key = ("intranet_server", "192.0.2.1", "192.0.2.2")
    requests = {key: ([100], [anchor])}
    flow = {
        "scenario": "harrison", "capture_host": "intranet_server",
        "source_address": "192.0.2.1", "destination_address": "192.0.2.2",
        "flow_id": "same", "stable_flow_id": "same-stable",
        "connection_start": 100.8, "observed_end": 101.1,
        "raw8": {"ip_protocol": 6, "resp_port": 443},
    }
    add_observation(flow, requests)
    add_observation(dict(flow, flow_id="same-other", stable_flow_id="other-stable"), requests)
    add_observation(dict(flow, connection_start=101.0, flow_id="next-second"), requests)
    add_observation(dict(flow, source_address="192.0.2.3", flow_id="other-peer"), requests)
    add_observation(dict(flow, capture_host="cloud_share", flow_id="other-host"), requests)
    add_observation(dict(flow, observed_end=None, flow_id="unbounded"), requests)
    assert anchor["candidate_flow_ids"] == [
        {"flow_id": "same", "stable_flow_id": "same-stable"},
        {"flow_id": "same-other", "stable_flow_id": "other-stable"},
    ]
    assert anchor["unbounded_candidate_flow_ids"][0]["flow_id"] == "unbounded"
    assert "label" not in anchor


def test_whole_connection_normal_review_rejects_unanchored_service_request():
    row = {
        "flow_id": "candidate", "stable_flow_id": "stable-candidate", "capture_host": "cloud_share",
        "raw8": dict.fromkeys(RAW8, 1.0), "raw8_complete": True, "conn_state": "SF",
        "flow_key_unique": True, "connection_start": 21.1, "observed_end": 22.2,
        "replay_export_time": 22.3, "after_input_eof": False, "during_shutdown": False,
    }
    anchors = [{"event_id": "affirmative", "control_annotation": False, "unique_flow": True}]
    known = {"author_annotation_lines": [], "affirmative_normal_event_ids": ["affirmative"]}
    unknown = {"author_annotation_lines": [], "affirmative_normal_event_ids": []}
    with_unknown = evaluate_connection(row, anchors, [known, unknown], 20, 30)
    assert with_unknown["review_status"] == "unresolved"
    assert with_unknown["exclusion_reasons"] == [
        "full_connection_has_some_requests_without_positive_normal_evidence"
    ]
    bound = evaluate_connection(row, anchors, [known], 20, 30)
    assert bound["review_status"] == "source_bound_normal_candidate"
    assert "label" not in bound


def test_http_request_identity_is_port_and_query_sensitive_without_raw_content():
    row = {
        "host": "Example.COM:8080", "uri": "/example?value=1", "id.resp_p": 8080,
    }
    assert get_http_digest(row) == http_request_digest("example.com", 8080, "/example", "value=1")
    assert get_http_digest({**row, "uri": "/example?value=2"}) != get_http_digest(row)
    assert get_http_digest({**row, "host": "u:p@Example.COM:8080"}) is None
