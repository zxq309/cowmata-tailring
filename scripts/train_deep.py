"""Compatibility shim: `python -m scripts.train_deep` == `cowmata train`."""

from __future__ import annotations

import sys

from cowmata.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
