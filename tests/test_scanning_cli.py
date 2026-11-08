"""Command routing without loading real data, preprocessing or models."""
from pathlib import Path

import pytest

from ids.context_cache import __main__ as cli


@pytest.mark.parametrize("command,arguments", [
    ("scanning-register", ["--round14-root", "old", "--checkpoint", "random.ckpt", "--output", "config.json"]),
    ("scanning-preflight", ["--config", "config.json", "--output", "preflight.json"]),
    ("lab-scanning-compare", ["--config", "config.json", "--output", "results"]),
    ("scanning-predict", ["--deployment", "bundle", "--input", "query.csv", "--output", "scores.csv"]),
    ("scanning-verify", ["--results", "results", "--output", "verified.json"]),
    ("scanning-pack", ["--run-root", "run", "--output", "result.zip"]),
])
def test_scanning_commands_route(command, arguments, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(cli, "scanning_command", lambda args: called.append(args) or {"ok": True})
    assert cli.main([command, *arguments]) == 0
    assert called[0].command == command
    assert isinstance(called[0].output, Path)
    assert '"ok": true' in capsys.readouterr().out


@pytest.mark.parametrize("command", ["scanning-preflight", "lab-scanning-compare"])
def test_lab_cli_cannot_run_synthetic_protocol(command, monkeypatch):
    from ids.context_cache import scanning_protocol
    monkeypatch.setattr(scanning_protocol, "load_scanning_protocol",
                        lambda _: ({"data_kind": "synthetic"}, None, None))
    args = cli.build_parser().parse_args([command, "--config", "x", "--output", "unused"])
    with pytest.raises(ValueError, match="requires frozen development"):
        cli.scanning_command(args)
