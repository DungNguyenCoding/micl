"""Compatibility wrapper for the canonical main.py entry point."""

from __future__ import annotations

import sys

import main


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv[1:1] = ["--dataset", "mnist"]
    main.main()
