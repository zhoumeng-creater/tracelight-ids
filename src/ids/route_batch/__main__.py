"""Laboratory batch entry point; registration and preflight never load real models."""
import argparse
import json

from ids.common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--round14-root", required=True)
    register.add_argument("--checkpoint", required=True)
    register.add_argument("--output", required=True)
    register.add_argument("--tabpfn-checkpoint")
    register.add_argument("--tabpfn-python")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--output")
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--retry-unavailable", action="store_true")
    worker = commands.add_parser("_worker")
    worker.add_argument("--config", required=True)
    worker.add_argument("--task", required=True)
    worker.add_argument("--output", required=True)
    cost_worker = commands.add_parser("_cost-worker")
    cost_worker.add_argument("--config", required=True)
    cost_worker.add_argument("--output", required=True)
    confirmation = commands.add_parser("confirm")
    confirmation.add_argument("--run-dir", required=True)
    confirmation.add_argument("--source-zip", required=True)
    confirmation.add_argument("--round13-prepared", required=True)
    confirmation.add_argument("--retry-failed", action="store_true")
    confirm_worker = commands.add_parser("_confirm-worker")
    confirm_worker.add_argument("--run-dir", required=True)
    confirm_worker.add_argument("--model-id", required=True)
    confirm_worker.add_argument("--retry-failed", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--run-dir", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--run-dir", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--include-models", action="store_true")
    for name in ("followup-preflight", "followup"):
        followup = commands.add_parser(name)
        followup.add_argument("--original-run", required=True)
        followup.add_argument("--output", required=True)
        followup.add_argument("--tabpfn-python", required=True)
        followup.add_argument("--tabpfn-checkpoint", required=True)
        followup.add_argument("--preparation-report")
        if name == "followup-preflight":
            followup.add_argument("--report")
        else:
            followup.add_argument("--resume", action="store_true")
    followup_worker = commands.add_parser("_followup-worker")
    followup_worker.add_argument("--run-dir", required=True)
    followup_worker.add_argument("--task", required=True)
    followup_worker.add_argument("--output", required=True)
    followup_pack = commands.add_parser("followup-pack")
    followup_pack.add_argument("--run-dir", required=True)
    followup_pack.add_argument("--output", required=True)
    full_replay = commands.add_parser("_followup-full-replay")
    full_replay.add_argument("--config", required=True)
    full_replay.add_argument("--output", required=True)
    replay = commands.add_parser("_replay-path-worker")
    replay.add_argument("--original-run", required=True)
    replay.add_argument("--task", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--build-path", choices=("original", "cpu"), required=True)
    replay.add_argument("--checkpoint")
    args = parser.parse_args()
    if args.command == "register":
        from .protocol import register
        result = register(args.round14_root, args.checkpoint, args.output,
                          tabpfn_checkpoint=args.tabpfn_checkpoint, tabpfn_python=args.tabpfn_python)
    elif args.command == "_cost-worker":
        from .cost_worker import run_cost
        result = run_cost(args.config, args.output)
    elif args.command == "_replay-path-worker":
        from .replay import replay_path_worker
        result = replay_path_worker(args.original_run, args.task, args.output,
                                    build_path=args.build_path, checkpoint_path=args.checkpoint)
    elif args.command == "_followup-full-replay":
        from .deployment_replay import run
        result = run(args.config, args.output)
    elif args.command in {"followup-preflight", "followup", "_followup-worker", "followup-pack"}:
        from . import followup
        if args.command == "_followup-worker":
            result = followup.worker(args.run_dir, args.task, args.output)
        elif args.command == "followup-pack":
            result = followup.pack(args.run_dir, args.output)
        else:
            options = {"tabpfn_python": args.tabpfn_python, "tabpfn_checkpoint": args.tabpfn_checkpoint,
                       "preparation_report": args.preparation_report}
            if args.command == "followup-preflight":
                result = followup.preflight(args.original_run, args.output, **options)
                if args.report:
                    write_json(args.report, result)
            else:
                result = followup.run(args.original_run, args.output, resume=args.resume, **options)
    elif args.command == "confirm":
        from .confirm_runner import run_confirmation
        result = run_confirmation(args.run_dir, args.source_zip, args.round13_prepared,
                                  retry_failed=args.retry_failed)
    elif args.command == "_confirm-worker":
        from .confirm_runner import confirm_worker
        result = confirm_worker(args.run_dir, args.model_id, retry_failed=args.retry_failed)
    else:
        from . import runner
        if args.command == "preflight":
            result = runner.preflight(args.config)
            if args.output:
                write_json(args.output, result)
        elif args.command == "run":
            result = runner.run(args.config, args.output, resume=args.resume, retry_unavailable=args.retry_unavailable)
        elif args.command == "_worker":
            result = runner.worker(args.config, args.task, args.output)
        elif args.command == "summarize":
            result = runner.summarize(args.run_dir)
        else:
            result = runner.pack(args.run_dir, args.output, include_models=args.include_models)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
