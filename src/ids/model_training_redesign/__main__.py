"""Laboratory execution and zero-fit input/archive checks for the fixed redesign."""
import argparse
import json
from pathlib import Path

from .protocol import specifications, tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("local/model-training-redesign-r30"))
    parser.add_argument("--r29-run", type=Path)
    parser.add_argument("--r26-run", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    modes = parser.add_mutually_exclusive_group()
    for name in ("plan-only", "inputs-only", "verify-only", "pack-only"):
        modes.add_argument("--" + name, action="store_true")
    modes.add_argument("--worker")
    modes.add_argument("--cost-worker")
    args = parser.parse_args()
    if args.plan_only:
        registration, design = specifications()
        print(json.dumps({"version": registration["version"], "tasks": tasks(design),
                          "max_fits": 6, "final_refits": 0, "roles": {
                              k: {f: v[f] for f in ("rows", "normal", "attack")}
                              for k, v in design["roles"].items()},
                          "selection": design["selection"], "model_execution": False}, indent=2))
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
        if args.r29_run is None or args.raw_dir is None:
            parser.error("--r29-run and --raw-dir are required; no model or dataset download is performed")
        result = runner.run(root, r29=args.r29_run, r26=args.r26_run, raw_dir=args.raw_dir,
                            threads=args.threads, device=args.device, inputs_only=args.inputs_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
