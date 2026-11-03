"""Lab entry point for the published Stage 5 frozen matrix."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tomllib

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RELEASE_FILES = (
    "workflows/stage5/run.py", "workflows/stage5/core.py", "workflows/stage5/reporting.py",
    "workflows/stage5/protocol.json", "docs/STAGE5_PROTOCOL.md", "docs/STAGE5_LAB.md",
)
OFFICIAL_ORIGINS = {
    "https://github.com/zhoumeng-creater/tracelight-ids",
    "git@github.com:zhoumeng-creater/tracelight-ids",
    "ssh://git@github.com/zhoumeng-creater/tracelight-ids",
}


def load_protocol() -> dict:
    """The public CLI has one fixed policy, independent of the launch directory."""
    config = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    if config.get("profile") != "research" or config.get("stage") != "stage5-frozen-evaluation-v1":
        raise ValueError("This entry point requires the published Stage 5 research policy")
    return config


def check_release(repo: Path) -> str:
    """Accept a clean, published evaluator, including an older main checkout for resume."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       encoding="utf-8", stderr=subprocess.PIPE).strip()

    try:
        origin = git("remote", "get-url", "origin").rstrip("/").removesuffix(".git")
        if origin not in OFFICIAL_ORIGINS:
            raise ValueError("Use the published TraceLight repository as origin")
        head = git("rev-parse", "--verify", "HEAD")
        git("merge-base", "--is-ancestor", head, "refs/remotes/origin/main")
        git("ls-files", "--error-unmatch", "--", *RELEASE_FILES)
        if git("status", "--porcelain", "--", *RELEASE_FILES):
            raise ValueError("Local Stage 5 evaluator or protocol differs from its published commit")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Cannot verify this Stage 5 release. Fetch the official main branch "
                         "and use a published checkout with all evaluator files.") from exc
    return head


def check_dependencies(repo: Path) -> None:
    """Verify runtime versions from the same project file bound by Stage 4."""
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        raise ValueError("Use the project Python 3.11 or 3.12 environment")
    project = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    mismatches = []
    for requirement in [*project["dependencies"], *project["optional-dependencies"]["torch"]]:
        name, expected = requirement.split("==")
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual.split("+", 1)[0] != expected:
            mismatches.append(f"{name}: expected {expected}, found {actual}")
    if mismatches:
        raise ValueError("Runtime dependency mismatch: " + ", ".join(mismatches))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the published Stage 5 calibration and final test matrix.")
    parser.add_argument("--stage4-work-dir", type=Path, required=True,
                        help="Original Stage 4 workspace containing experiment/ and data/cicids-prepared/")
    parser.add_argument("--unsw-data-dir", type=Path, required=True, help="Original prepared UNSW directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate Stage 5 output directory")
    parser.add_argument("--resume", action="store_true", help="Verify and continue the same frozen evaluation")
    args = parser.parse_args(argv)
    work, unsw, output = (p.resolve() for p in (args.stage4_work_dir, args.unsw_data_dir, args.output_dir))
    if output.is_relative_to(work) or work.is_relative_to(output):
        raise ValueError("Choose a Stage 5 output directory separate from the Stage 4 workspace")
    experiment = work / "experiment"
    datasets = {"unsw": unsw, "cicids": work / "data" / "cicids-prepared"}
    for path in [experiment / "STAGE4_COMPLETE.json", *(d / "manifest.json" for d in datasets.values())]:
        if not path.is_file():
            raise ValueError(f"Required original Stage 4 or prepared input is missing: {path}")

    release = check_release(REPO)
    check_dependencies(REPO)
    config = load_protocol()
    # Prefer the checked-out sources over globally installed packages when run as a script.
    sys.path[:0] = [str(REPO / "src"), str(REPO)]
    from workflows.stage5.core import run_evaluation
    from workflows.stage5.reporting import finalize

    print(f"Stage 5 release {release}. Verifying the frozen matrix.", flush=True)
    run_evaluation(experiment, datasets, output, config, resume=args.resume)
    package = finalize(output, config)
    print(f"Report: {output / 'reports' / 'index.html'}")
    print(f"Results: {output / 'reports' / 'summary.json'}")
    print(f"Return: {package} and {package}.sha256")


if __name__ == "__main__":
    main()
