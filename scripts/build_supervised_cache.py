"""Compatibility shim: `python -m scripts.build_supervised_cache` == `cowmata build-cache`."""

from __future__ import annotations

import sys

from cowmata.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-cache", *sys.argv[1:]]))
