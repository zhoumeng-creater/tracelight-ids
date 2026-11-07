"""Native input contracts and a random-tiny-checkpoint CLI round trip.

No pretrained weights or network are used. Optional official-source absence
skips only the architecture round trip, not the native input contract checks.
"""

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ids.context_cache import __main__ as cli  # noqa: E402


def native_payload(role="support"):
    payload = {
        "format": cli.INPUT_FORMAT,
        "role": role,
        "data_kind": "synthetic",
        "X": torch.arange(24, dtype=torch.float32).reshape(1, 8, 3) / 10,
        "feature_names": ["synthetic_a", "synthetic_b", "synthetic_c"],
        "preprocessing_id": "synthetic_identity_support_only",
        "label_mapping": {"normal": 0, "attack": 1},
    }
    if role == "support":
        payload["y"] = torch.tensor([[0, 1] * 4])
    return payload


def test_native_input_preserves_feature_order_and_label_direction(tmp_path):
    path = tmp_path / "support.pt"
    payload = native_payload()
    torch.save(payload, path)
    loaded, contract = cli._read_input(path, "support")
    assert torch.equal(loaded["X"], payload["X"])
    assert contract == {name: payload[name] for name in
                        ("feature_names", "preprocessing_id", "label_mapping")}


@pytest.mark.parametrize("change, expected", [
    ({"role": "final_test"}, "role=support"),
    ({"data_kind": "final_test"}, "synthetic/development"),
    ({"feature_names": ["b", "b", "c"]}, "feature_names"),
    ({"feature_names": ["a", "b"]}, "feature_names"),
    ({"preprocessing_id": ""}, "preprocessing_id"),
    ({"label_mapping": {"normal": 1, "attack": 0}}, "normal=0, attack=1"),
    ({"y": torch.tensor([[0, 2] * 4])}, "only 0/1"),
])
def test_native_input_rejects_incompatible_contract(tmp_path, change, expected):
    payload = native_payload()
    payload.update(change)
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=expected):
        cli._read_input(path, "support")


def test_query_rejects_labels_even_when_other_contract_fields_match(tmp_path):
    payload = native_payload("query")
    payload["y"] = torch.tensor([[0, 1] * 4])
    path = tmp_path / "query_with_labels.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="no labels"):
        cli._read_input(path, "query")


def test_compile_predict_cli_with_random_tiny_official_checkpoint(tmp_path, monkeypatch, capsys):
    checkout = Path(__file__).resolve().parents[1] / "local" / "vendor" / "tabicl"
    if not (checkout / "src" / "tabicl" / "_model" / "tabicl.py").is_file():
        pytest.skip("Pinned optional TabICLv2 source absent; no source or checkpoint is downloaded")
    monkeypatch.syspath_prepend(str(checkout / "src"))
    from ids.context_cache.upstream import require_pinned_tabicl

    require_pinned_tabicl()
    from tabicl._model.tabicl import TabICL

    config = {
        "max_classes": 10, "embed_dim": 8, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 4, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2,
        "icl_num_blocks": 2, "icl_nhead": 2, "ff_factor": 2, "dropout": 0.0,
        "zero_init": False, "col_feature_group": "same", "col_feature_group_size": 3,
        "col_target_aware": True, "col_ssmax": "qassmax-mlp-elementwise",
        "icl_ssmax": "qassmax-mlp-elementwise",
    }
    checkpoint = tmp_path / "random_tiny_NOT_PRETRAINED.ckpt"
    with torch.random.fork_rng():
        torch.manual_seed(7103)
        model = TabICL(**config).eval()
    torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    support_path, query_path = tmp_path / "support.pt", tmp_path / "query.pt"
    torch.save(native_payload(), support_path)
    query = native_payload("query")
    query["X"] = torch.linspace(-1, 1, 15).reshape(1, 5, 3)
    torch.save(query, query_path)
    output = tmp_path / "compiled"
    prediction = tmp_path / "predictions.pt"
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        assert cli.main(["--tabicl-source", str(checkout), "compile", "--checkpoint", str(checkpoint),
                         "--checkpoint-sha256", checkpoint_hash, "--support", str(support_path),
                         "--slots", "2", "--query-chunk-size", "3", "--output", str(output)]) == 0
        report = json.loads((output / "compile_report.json").read_text(encoding="utf-8"))
        assert report["stage"] == "cache_compilation_only"
        assert report["data_kind"] == "synthetic"
        assert report["compilation"]["queries_are_full_support_attention_queries"]
        assert (output / "full.pt").is_file() and (output / "compact.pt").is_file()
        assert cli.main(["predict", "--bundle", str(output / "compact.pt"), "--queries", str(query_path),
                         "--output", str(prediction)]) == 0
        predictions = torch.load(prediction, weights_only=True)
        assert predictions["logits"].shape == (1, 5, 2)
        assert torch.isfinite(predictions["logits"]).all()
        torch.testing.assert_close(predictions["scores"],
                                   predictions["logits"][..., 1] - predictions["logits"][..., 0])
        assert predictions["score_kind"] == "attack_minus_normal_logit"
        assert predictions["input_contract"] == report["input_contract"]

        # Equal feature count is insufficient: order is part of the contract.
        query["feature_names"] = list(reversed(query["feature_names"]))
        torch.save(query, query_path)
        wrong_output = tmp_path / "wrong_order.pt"
        with pytest.raises(ValueError, match="contract differs"):
            cli.main(["predict", "--bundle", str(output / "compact.pt"), "--queries", str(query_path),
                      "--output", str(wrong_output)])
        assert not wrong_output.exists()
    finally:
        torch.set_num_threads(previous_threads)
    assert "uncalibrated_native_prediction" in capsys.readouterr().out
