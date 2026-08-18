"""Compatibility shim: `python -m scripts.mine_candidates` == `cowmata mine`."""

from __future__ import annotations

import sys

from cowmata.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["mine", *sys.argv[1:]]))
