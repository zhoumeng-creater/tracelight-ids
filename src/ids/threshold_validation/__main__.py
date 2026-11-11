"""Command entry; keep process-pool workers importable in Windows spawn mode."""

from .runner import main

if __name__ == "__main__":
    main()
