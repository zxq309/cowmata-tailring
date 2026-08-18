"""Compatibility shim: `python -m scripts.build_feature_table` == `cowmata build-features`."""

from __future__ import annotations

import sys

from cowmata.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-features", *sys.argv[1:]]))
