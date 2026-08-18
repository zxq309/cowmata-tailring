"""Compatibility shims for artefacts serialised by the 20260818 package.

``weights/deploy/gbdt_full.joblib`` was written while the algorithm core lived
in a package called ``cattle_imu``.  Pickle stores the *module path* of every
class it serialises, so unpickling that bundle looks for
``cattle_imu.gbdt.BinaryBooster`` and fails once the package is renamed.

The 20260818 baseline solved this by keeping the whole ``cattle_imu`` package
alive next to ``cowmata`` - two top-level packages doing one job, which is the
main reason the repository had three parallel concepts.  Twenty lines of module
aliasing solve it instead, and the alias is installed automatically on import of
:mod:`cowmata`, so nothing at a call site has to remember.

Re-serialising the bundle under the new path is the eventual fix.  It is not
done here because the deployed artefact is byte-identical to the one whose
predictions were verified, and quietly rewriting it would spend that verification.
"""

from __future__ import annotations

import sys
import types

LEGACY_ROOT = "cattle_imu"

#: Old module path -> new module path.
MODULE_ALIASES: dict[str, str] = {
    "cattle_imu": "cowmata",
    "cattle_imu.gbdt": "cowmata.gbdt",
    "cattle_imu.io": "cowmata.io",
    "cattle_imu.features": "cowmata.features",
    "cattle_imu.metrics": "cowmata.metrics",
    "cattle_imu.model": "cowmata.models",
    "cattle_imu.dataset": "cowmata.dataset",
    "cattle_imu.preprocessing": "cowmata.preprocessing",
    "cattle_imu.annotations": "cowmata.labels",
    "cattle_imu.state_machine": "cowmata.postprocess",
    "cattle_imu.amp": "cowmata.runtime",
    "cattle_imu.load_control": "cowmata.runtime",
}


def install_legacy_aliases() -> list[str]:
    """Make ``cattle_imu.*`` importable as an alias of ``cowmata.*``.

    Modules are aliased lazily: a placeholder is registered in ``sys.modules``
    and the real module is imported only when something actually touches it, so
    importing :mod:`cowmata` never drags in torch through the alias table.
    """

    installed: list[str] = []
    for legacy, target in MODULE_ALIASES.items():
        if legacy in sys.modules:
            continue
        sys.modules[legacy] = _LazyAlias(legacy, target)
        installed.append(legacy)
    return installed


class _LazyAlias(types.ModuleType):
    """A module object that imports its target on first attribute access."""

    def __init__(self, name: str, target: str) -> None:
        super().__init__(name)
        self.__dict__["_cowmata_target"] = target
        self.__dict__["_cowmata_loaded"] = None

    def _load(self) -> types.ModuleType:
        loaded = self.__dict__["_cowmata_loaded"]
        if loaded is None:
            import importlib

            loaded = importlib.import_module(self.__dict__["_cowmata_target"])
            self.__dict__["_cowmata_loaded"] = loaded
        return loaded

    def __getattr__(self, item: str):
        return getattr(self._load(), item)

    def __dir__(self) -> list[str]:
        return dir(self._load())
