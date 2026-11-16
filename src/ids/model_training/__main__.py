"""python -m ids.model_training: laboratory run and offline evidence tools."""
import argparse
import json
from pathlib import Path

from .protocol import specifications, tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("local/model-training-r29"))
    parser.add_argument("--r26-run", type=Path)
    parser.add_argument("--recovered-training", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    modes = parser.add_mutually_exclusive_group()
    for name in ("plan-only", "inputs-only", "verify-only", "pack-only"):
        modes.add_argument("--" + name, action="store_true")
    modes.add_argument("--worker")
    modes.add_argument("--cost-worker")
    args = parser.parse_args()
    if args.plan_only:
        registration, design = specifications()
        print(json.dumps({"registration": registration, "design": design, "tasks": tasks(design)}, indent=2))
        return
    from . import runner
    root = args.output_dir.resolve()
    if args.worker:
        runner.train_task(root, args.worker)
        return
    if args.cost_worker:
        runner.cost_worker(root, args.cost_worker)
        return
    if args.verify_only or args.pack_only:
        from ids.flow_history.execution import run_lock
        with run_lock(root):
            result = runner.pack(root) if args.pack_only else runner.verify(root)
    else:
        if args.r26_run is None:
            parser.error("--r26-run is required for input preparation or laboratory training")
        result = runner.run(root, r26=args.r26_run, recovered=args.recovered_training,
                            raw_dir=args.raw_dir, threads=args.threads, device=args.device,
                            inputs_only=args.inputs_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
