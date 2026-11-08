"""Command routing without loading real data, preprocessing or models."""
from pathlib import Path

import pytest

from ids.context_cache import __main__ as cli


@pytest.mark.parametrize("command,arguments", [
    ("label-register", ["--round14-root", "old", "--checkpoint", "random.ckpt", "--output", "config.json"]),
    ("label-preflight", ["--config", "config.json", "--output", "preflight.json"]),
    ("lab-label-compare", ["--config", "config.json", "--output", "results"]),
    ("label-predict", ["--deployment", "bundle", "--input", "query.csv", "--output", "scores.csv"]),
    ("label-verify", ["--results", "results", "--output", "verified.json"]),
    ("label-pack", ["--run-root", "run", "--output", "result.zip"]),
])
def test_label_commands_route(command, arguments, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(cli, "label_command", lambda args: called.append(args) or {"ok": True})
    assert cli.main([command, *arguments]) == 0
    assert called[0].command == command
    assert isinstance(called[0].output, Path)
    assert '"ok": true' in capsys.readouterr().out


@pytest.mark.parametrize("command", ["label-preflight", "lab-label-compare"])
def test_lab_cli_cannot_run_synthetic_protocol(command, monkeypatch):
    from ids.context_cache import label_protocol
    monkeypatch.setattr(label_protocol, "load_label_protocol",
                        lambda _: ({"data_kind": "synthetic"}, None, None))
    args = cli.build_parser().parse_args([command, "--config", "x", "--output", "unused"])
    with pytest.raises(ValueError, match="requires frozen development"):
        cli.label_command(args)
