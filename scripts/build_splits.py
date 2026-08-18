"""Compatibility shim: `python -m scripts.build_splits` == `cowmata make-splits`."""

from __future__ import annotations

import sys

from cowmata.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["make-splits", *sys.argv[1:]]))
