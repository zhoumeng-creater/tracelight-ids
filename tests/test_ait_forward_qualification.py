import io
import zipfile

import pytest

from scripts.ait_forward import qualification


class RangeResponse(io.BytesIO):
    def __init__(self, payload, status, content_range):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range}


def test_archive_index_and_manifest_do_not_request_capture_payload(monkeypatch):
    buffer = io.BytesIO()
    capture = "gather/intranet_server/logs/suricata/log.pcap.1"
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(capture, b"x" * 100_000)
        archive.writestr(
            "dataset.yaml",
            "name: processed_harrison_scenario\n"
            "start: '2022-02-04T00:00:00'\nend: '2022-02-09T00:00:00'\n",
        )
        archive.writestr("labels/intranet_server/label.json", "{}")
        archive.writestr("gather/internal_employee_1/logs/sm.log", "log")
        archive.writestr("gather/intranet_server/logs/apache2/access.log", "")
        archive.writestr("gather/intranet_server/logs/apache2/vhost-access.log.1", "log")
        archive.writestr("gather/cloud_share/logs/apache2/other_vhosts_access.log.1.gz", "log")
    contents = buffer.getvalue()
    capture_end = 100_030 + len(capture)

    def open_range(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        assert start >= capture_end
        return RangeResponse(contents[start:end + 1], 206, f"bytes {start}-{end}/{len(contents)}")

    monkeypatch.setattr(qualification, "urlopen", open_range)
    with qualification.RangeReader("harrison_no-pcaps.zip", len(contents)) as remote:
        with zipfile.ZipFile(remote) as archive:
            indexed = qualification.summarize_index(archive)
            dates = qualification.read_time_manifest(archive)
        assert remote.receipts
        assert all(len(entry["sha256"]) == 64 for entry in remote.receipts)

    assert indexed["service_members"]["intranet_server"]["network_capture_member_count"] == 1
    assert indexed["service_members"]["intranet_server"]["label_member_count"] == 1
    assert indexed["service_members"]["intranet_server"]["access_log_member_count"] == 2
    assert indexed["service_members"]["intranet_server"]["nonempty_access_log_member_count"] == 1
    assert indexed["service_members"]["cloud_share"]["nonempty_access_log_member_count"] == 1
    assert indexed["normal_actor_log_member_count"] == 1
    assert indexed["nonempty_normal_actor_log_member_count"] == 1
    assert dates["start"] == "2022-02-04T00:00:00"
    assert dates["end"] == "2022-02-09T00:00:00"
    assert dates["timezone_reported"] is False
    assert set(dates) == {"start", "end", "path", "size", "zip_crc32", "sha256", "timezone_reported"}


def test_range_reader_rejects_full_response_before_reading(monkeypatch):
    class UnboundedResponse(RangeResponse):
        def read(self, size=-1):
            pytest.fail("Qualification must not read HTTP 200 archive bodies")

    monkeypatch.setattr(
        qualification, "urlopen", lambda request, timeout: UnboundedResponse(b"", 200, "")
    )
    with qualification.RangeReader("harrison.zip", 10) as remote:
        with pytest.raises(ValueError, match="did not return the requested range"):
            remote.read(4)
        assert remote.tell() == 0
        assert not remote.receipts
