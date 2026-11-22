"""A process-isolated registered training trajectory; only F is parsed."""
from pathlib import Path
import argparse
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from scripts.comprehensive_batch import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--coverage-results", type=Path)
    parser.add_argument("--gpu-workers", type=int, required=True)
    parser.add_argument("--cpu-workers", type=int, required=True)
    args = parser.parse_args()
    registration, plan, rows, processor = run.prepare()
    output = args.output.resolve()
    receipt = run.read(output / "source_receipt.json")
    if (receipt["registration_sha256_lf"] != run.text_digest(REPO / run.DIRECTORY / "registration.json")
            or receipt["plan_sha256"] != run.data.json_digest(plan)):
        raise ValueError("Worker code or fixed plan differs from coordinator")
    jobs = {job["job_id"]: job for job in [*plan["hgb_jobs"], *plan["tabm_jobs"]]}
    job = jobs[args.job_id]
    stop = output / "execution_stop.request"
    if stop.exists():
        raise InterruptedError("Coordinator requested a checkpoint stop")
    run.execute_job(output, rows, processor, job, job["family"], resume=args.resume,
                    coverage_results=args.coverage_results, stop_path=stop,
                    execution_context={"mode": "independent_processes", "gpu_workers": args.gpu_workers,
                                       "cpu_workers": args.cpu_workers, "threads": 4,
                                       "wall_time_includes_resource_contention": True})


if __name__ == "__main__":
    main()
