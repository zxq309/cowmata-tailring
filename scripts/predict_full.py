"""Compatibility shim: `python -m scripts.predict_full` == `cowmata predict`."""

from __future__ import annotations

import sys

from cowmata.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["predict", *sys.argv[1:]]))
