import argparse
from pathlib import Path
import traceback

import psutil

from ids.common import utc_now, write_json
from .execution import ResourceLog, own_windows_workers, pack, run_lock


def main():
    parser = argparse.ArgumentParser(description="Run or resume the two registered flow-history comparisons")
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="optional official CSV or ZIP when original data was relocated")
    parser.add_argument("--threads", type=int, default=min(24, psutil.cpu_count(logical=False) or 2))
    args = parser.parse_args()
    root, original = args.output.resolve(), args.original_run.resolve()
    if args.threads < 2:
        parser.error("this laboratory batch requires at least two CPU threads")
    if root == original or root in original.parents or original in root.parents:
        parser.error("output and original-run must be separate directories")
    if not original.is_dir():
        parser.error(f"registered previous run is missing: {original}")
    if root.exists() and not (root / "run.lock").is_file() and any(root.iterdir()):
        parser.error("an existing nonempty output must belong to this experiment")
    from .pipeline import run
    with run_lock(root):
        try:
            own_windows_workers()
            with ResourceLog(root) as monitor:
                run(root, original, args.source.resolve() if args.source else None, args.threads, monitor)
        except Exception:
            write_json(root / "last_invocation.json", {"status": "failed", "at": utc_now(),
                                                      "error": traceback.format_exc()})
            pack(root)
            raise
        write_json(root / "last_invocation.json", {"status": "complete", "at": utc_now()})
        pack(root)


if __name__ == "__main__":
    main()
