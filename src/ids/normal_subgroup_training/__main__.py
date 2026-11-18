"""Explicit preflight, registered laboratory run, workers and offline audit."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "prepare-only", "run", "verify-only", "pack-only", "plan"):
        modes.add_argument("--" + mode, action="store_true")
    modes.add_argument("--worker")
    modes.add_argument("--cost-worker")
    parser.add_argument("--r30-archive")
    parser.add_argument("--output")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.plan and not args.output:
        parser.error("--output is required")
    if (args.preflight or args.prepare_only or args.run) and not args.r30_archive:
        parser.error("--r30-archive is required")
    if args.plan:
        from .protocol import specifications, tasks
        registration, design = specifications()
        result = {"registration": registration, "tasks": tasks(design), "model_execution": False}
    elif args.preflight:
        from .preflight import run
        result = run(args.r30_archive, args.output)
    else:
        from . import runner
        root = Path(args.output).resolve()
        if args.worker:
            result = runner.train_task(root, args.worker)
        elif args.cost_worker:
            result = runner.cost_worker(root, args.cost_worker)
        elif args.verify_only:
            result = runner.verify(root)
        elif args.pack_only:
            result = runner.pack(root)
        else:
            result = runner.run(root, archive=args.r30_archive, threads=args.threads,
                                device=args.device, prepare_only=args.prepare_only)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
