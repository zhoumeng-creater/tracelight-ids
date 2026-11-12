from ids.common import sha256_file, write_json
from ids.response_evidence import controls


def test_reuse_requires_frozen_sources_environment_and_files(tmp_path, monkeypatch):
    old = tmp_path / "old"
    path = old / "models/frozen/hgb-raw.joblib"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"an archived synthetic model")
    name = path.relative_to(old).as_posix()
    fixture = tmp_path / "reference"
    write_json(
        fixture / "r22-model-reference.json",
        {
            "inputs": {"P": "abc"},
            "runtime": {"python": "fixture"},
            "artifacts": {"hgb-raw": {name: sha256_file(path)}},
        },
    )
    monkeypatch.setattr(controls, "FOLDER", fixture)
    monkeypatch.setenv("TRACELIGHT_R22_OUTPUT", str(old))
    reg = {"sources": [{"name": "P", "sha256": "abc", "role": "P"}]}
    assert controls.previous_output(reg, {"python": "fixture"}) == str(old)
    assert controls.previous_output(reg, {"python": "changed"}) is None
    (copied,) = controls.reuse(tmp_path / "run", old, "hgb", "raw")
    assert copied.read_bytes() == path.read_bytes()
    path.write_bytes(b"modified")
    assert controls.previous_output(reg, {"python": "fixture"}) is None
