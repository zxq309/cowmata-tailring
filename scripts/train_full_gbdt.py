"""Compatibility shim: `python -m scripts.train_full_gbdt` == `cowmata train-gbdt`."""

from __future__ import annotations

import sys

from cowmata.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train-gbdt", *sys.argv[1:]]))
