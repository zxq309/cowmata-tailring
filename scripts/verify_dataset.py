"""Compatibility shim: `python -m scripts.verify_dataset` == `cowmata check-data`."""

from __future__ import annotations

import sys

from cowmata.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["check-data", *sys.argv[1:]]))
