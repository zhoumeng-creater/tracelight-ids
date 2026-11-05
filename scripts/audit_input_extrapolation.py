"""Zero-training frozen screen audit; requires the original lab prepared data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ids.input_extrapolation import run_input_extrapolation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure-cpu", action="store_true", help="mixed workload timing on the lab host")
    args = parser.parse_args()
    result = run_input_extrapolation(args.screen_dir, args.data_dir, args.output, measure_cpu=args.measure_cpu)
    print(f"{result['status']}: {result['n_models']} frozen models, new model fits=0")


if __name__ == "__main__":
    main()
