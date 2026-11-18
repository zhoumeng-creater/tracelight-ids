"""The package currently exposes zero-update preflight only, never training."""
import argparse

from .preflight import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--r30-archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.r30_archive, args.output)


if __name__ == "__main__":
    main()
