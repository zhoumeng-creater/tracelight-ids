"""Official TabM process-isolation parity; generated rows and pytest tempdirs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

pytest.importorskip("tabm")
pytest.importorskip("rtdl_num_embeddings")

from ids.recent_baselines import _numeric_bins, _parameters, make_tabm_network
from scripts.comprehensive_batch import training


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29)


def _worker(output, seed, device, barrier):
    """Run the same two-step official-model job in one isolated interpreter."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    actual = training._device(device, 4)
    rng = np.random.Generator(np.random.PCG64(941))
    numeric = rng.normal(size=(48, 5)).astype(np.float32)
    cardinalities = [6, 2, 2, 2, 2, 2, 2, 2]
    categorical = np.column_stack([rng.integers(c, size=48, dtype=np.int64) for c in cardinalities])
    labels = np.array([0] * 32 + [1] * 16, dtype=np.float32)
    groups = {"references": [{"id": "r1", "indices": list(range(16))},
                              {"id": "r2", "indices": list(range(16, 32))}],
              "targets": [{"id": f"g{i}", "indices": list(range(i, i + 8))} for i in range(0, 32, 8)],
              "attacks": list(range(32, 48))}
    parameters = _parameters(training.FIXED_MODEL)
    bins, _ = _numeric_bins(numeric, parameters)
    spec = dict(training.FIXED_TRAINING, batch_size=16, draws_per_class=8,
                score_batch_size=32, report_every=1, recovery_every=1)
    job = {"job_id": f"generated_process_s{seed}", "arm": "S_alignment_group", "seed": seed,
           "learning_rate": .002, "auxiliary_weight": 1., "optimizer_steps": 2, "checkpoints": [1, 2]}
    with training.scoring_arithmetic():
        training._seed(seed, actual)
        network = make_tabm_network(5, cardinalities, parameters, numeric_bins=bins).to(actual)
        if barrier is not None:
            barrier = Path(barrier)
            (barrier / f"ready_{seed}.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf8")
            deadline = time.monotonic() + 120
            while not (barrier / "release").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Synthetic process barrier was not released")
                time.sleep(.01)
        started = time.time_ns()
        result = training.train_loop(network, numeric, categorical, labels, groups,
                                     job=job, device=actual, output_dir=output / "job", training_spec=spec)
        finished = time.time_ns()
        scores = np.load(output / "job" / result["snapshots"][-1]["F_scores"])["scores"]
        np.save(output / "scores.npy", scores)
        numerical = {key: result[key] for key in (
            "state_sha256", "initial_state_sha256", "initial_dropout_rng_sha256", "final_dropout_rng_sha256",
            "sampler_draw_sha256", "sampling")}
        numerical["scores_sha256"] = hashlib.sha256(scores.tobytes()).hexdigest()
        numerical["checkpoint_states"] = [item["state_sha256"] for item in result["snapshots"]]
        payload = {"seed": seed, "device": str(actual), "pid": os.getpid(),
                   "num_threads": torch.get_num_threads(), "started_ns": started, "finished_ns": finished,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"], "numerical": numerical}
        (output / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf8")


def _start(output, seed, device, barrier=None):
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
               "--output", str(output), "--seed", str(seed), "--device", device]
    if barrier is not None:
        command += ["--barrier", str(barrier)]
    return subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf8",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def _finish(process):
    output, _ = process.communicate(timeout=120)
    assert process.returncode == 0, output


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_serial_and_concurrent_official_jobs_have_identical_numerical_state(tmp_path, device):
    processes = []
    try:
        for seed in SEEDS:
            process = _start(tmp_path / f"serial_s{seed}", seed, device)
            processes.append(process)
            _finish(process)
        barrier = tmp_path / "barrier"
        barrier.mkdir()
        concurrent = [_start(tmp_path / f"parallel_s{seed}", seed, device, barrier) for seed in SEEDS]
        processes.extend(concurrent)
        deadline = time.monotonic() + 120
        while not all((barrier / f"ready_{seed}.json").exists() for seed in SEEDS):
            for process in concurrent:
                if process.poll() is not None:
                    _finish(process)
                    pytest.fail("Synthetic worker exited before the shared start barrier")
            assert time.monotonic() < deadline, "Synthetic workers did not reach the shared start barrier"
            time.sleep(.025)
        (barrier / "release").write_text("start both generated jobs", encoding="utf8")
        for process in concurrent:
            _finish(process)
        intervals, pids, seed_states = [], [], []
        for seed in SEEDS:
            serial_dir, parallel_dir = tmp_path / f"serial_s{seed}", tmp_path / f"parallel_s{seed}"
            serial = json.loads((serial_dir / "result.json").read_text(encoding="utf8"))
            parallel = json.loads((parallel_dir / "result.json").read_text(encoding="utf8"))
            assert serial["numerical"] == parallel["numerical"]
            np.testing.assert_array_equal(np.load(serial_dir / "scores.npy"), np.load(parallel_dir / "scores.npy"))
            for record in (serial, parallel):
                assert record["num_threads"] == 4 and record["deterministic_algorithms"]
                assert record["tf32"] is False and record["cublas_workspace"] == ":4096:8"
                assert record["pid"] != os.getpid()
            intervals.append((parallel["started_ns"], parallel["finished_ns"]))
            pids.append(parallel["pid"])
            seed_states.append(serial["numerical"]["initial_state_sha256"])
        assert len(set(pids)) == 2 and len(set(seed_states)) == 2
        assert max(start for start, _ in intervals) < min(end for _, end in intervals)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--barrier")
    arguments = parser.parse_args()
    _worker(arguments.output, arguments.seed, arguments.device, arguments.barrier)
