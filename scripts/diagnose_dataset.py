"""Compatibility shim: `python -m scripts.diagnose_dataset` == `cowmata diagnose`."""

from __future__ import annotations

import sys

from cowmata.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["diagnose", *sys.argv[1:]]))
