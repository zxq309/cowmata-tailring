"""Gradient-boosting backend with a single interface and a GPU-first policy.

Backend order: XGBoost (``device='cuda'``) -> LightGBM -> scikit-learn's
``HistGradientBoostingClassifier``.  The chosen backend is always recorded in
the report so a run can be reproduced.

``tree_method='gpu_hist'`` is deprecated in XGBoost 2.x; the current spelling
is ``tree_method='hist'`` together with ``device='cuda'``.

Unchanged from 20260818 except for its module path.  The deployed
``gbdt_full.joblib`` contains pickled ``BinaryBooster`` instances that name
``cattle_imu.gbdt``; :mod:`cowmata.compat` installs an alias so the existing
artefact keeps loading without being re-trained or re-serialised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BoosterConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    device: str = "cuda"
    random_state: int = 20260819
    extra: dict[str, Any] = field(default_factory=dict)


def available_backends() -> list[str]:
    names: list[str] = []
    try:  # pragma: no cover - depends on install
        import xgboost  # noqa: F401

        names.append("xgboost")
    except Exception:
        pass
    try:  # pragma: no cover
        import lightgbm  # noqa: F401

        names.append("lightgbm")
    except Exception:
        pass
    names.append("sklearn")
    return names


class BinaryBooster:
    """Binary classifier returning calibrated-ish probabilities in ``[0, 1]``."""

    def __init__(self, config: BoosterConfig | None = None, backend: str | None = None) -> None:
        self.config = config or BoosterConfig()
        options = available_backends()
        if backend is not None and backend not in options:
            raise ValueError(f"backend {backend!r} unavailable; installed: {options}")
        self.backend = backend or options[0]
        self.model: Any = None
        self.n_features: int | None = None

    # ------------------------------------------------------------------
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "BinaryBooster":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        self.n_features = int(x.shape[1])
        positive = int(y.sum())
        negative = int(y.size - positive)
        scale = max(negative, 1) / max(positive, 1)
        cfg = self.config

        if self.backend == "xgboost":  # pragma: no cover - requires GPU install
            import xgboost as xgb

            params = dict(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample,
                min_child_weight=cfg.min_child_weight,
                reg_lambda=cfg.reg_lambda,
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                device=cfg.device,
                random_state=cfg.random_state,
                scale_pos_weight=min(scale, 1000.0),
                **cfg.extra,
            )
            self.model = xgb.XGBClassifier(**params)
            fit_kwargs: dict[str, Any] = {}
            if eval_set is not None:
                fit_kwargs["eval_set"] = [(np.asarray(eval_set[0], np.float32), np.asarray(eval_set[1], np.int32))]
                fit_kwargs["verbose"] = False
            self.model.fit(x, y, sample_weight=sample_weight, **fit_kwargs)
        elif self.backend == "lightgbm":  # pragma: no cover
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                subsample=cfg.subsample,
                subsample_freq=1,
                colsample_bytree=cfg.colsample,
                min_child_weight=cfg.min_child_weight,
                reg_lambda=cfg.reg_lambda,
                scale_pos_weight=min(scale, 1000.0),
                random_state=cfg.random_state,
                verbose=-1,
                **cfg.extra,
            )
            self.model.fit(x, y, sample_weight=sample_weight)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            weight = sample_weight
            if weight is None:
                weight = np.where(y == 1, min(scale, 1000.0), 1.0).astype(np.float64)
            self.model = HistGradientBoostingClassifier(
                max_iter=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                l2_regularization=cfg.reg_lambda,
                min_samples_leaf=max(int(cfg.min_child_weight), 1),
                random_state=cfg.random_state,
            )
            self.model.fit(x, y, sample_weight=weight)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=np.float32)
        proba = self.model.predict_proba(x)
        return np.asarray(proba, dtype=np.float64)[:, 1]

    def feature_importance(self, names: list[str]) -> list[dict[str, object]]:
        importance = getattr(self.model, "feature_importances_", None)
        if importance is None:
            return []
        values = np.asarray(importance, dtype=np.float64)
        order = np.argsort(values)[::-1]
        return [
            {"feature": names[int(i)], "importance": float(values[int(i)])}
            for i in order
            if values[int(i)] > 0
        ]
