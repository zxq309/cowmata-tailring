"""Late fusion of the deep branch and the hand-crafted-feature branch.

Why late fusion and not concatenating features into the TCN input: with two or
three branches you can *see* which one carries each behaviour.  Early fusion
gives one number and no diagnosis, which is the wrong trade at this sample
size.

The fused score is a weighted average in logit space:

    z = sum_k w_k * logit(p_k),   w_k >= 0,  sum_k w_k = 1
    p = sigmoid(z)

Only ``len(branches) - 1`` free parameters are fitted, on the *validation*
split, by scanning the simplex on a coarse grid.  With so few parameters this
cannot meaningfully overfit, and it degrades to "use the best single branch"
when one branch is useless.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import average_precision_score

EPSILON = 1e-6


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=np.float64)))


def simplex_grid(dimension: int, steps: int = 10) -> list[tuple[float, ...]]:
    """All weight vectors on a ``steps``-resolution simplex."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if dimension == 1:
        return [(1.0,)]
    points: list[tuple[float, ...]] = []
    for combo in itertools.product(range(steps + 1), repeat=dimension - 1):
        used = sum(combo)
        if used > steps:
            continue
        weights = tuple(value / steps for value in combo) + ((steps - used) / steps,)
        points.append(weights)
    return points


def fit_fusion_weights(
    branch_probabilities: dict[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    steps: int = 10,
) -> dict[str, object]:
    """Fit fusion weights by maximising average precision on the given split."""

    names = sorted(branch_probabilities)
    if not names:
        raise ValueError("no branches supplied")
    stacked = np.stack([logit(branch_probabilities[name]) for name in names], axis=1)
    y = np.asarray(target).astype(np.int8)
    if mask is not None:
        keep = np.asarray(mask).astype(bool)
        stacked = stacked[keep]
        y = y[keep]
    if y.size == 0 or np.unique(y).size < 2:
        uniform = {name: 1.0 / len(names) for name in names}
        return {"weights": uniform, "average_precision": None, "status": "degenerate_split"}

    best_weights = None
    best_score = -np.inf
    per_branch = {}
    for index, name in enumerate(names):
        per_branch[name] = float(average_precision_score(y, stacked[:, index]))
    for weights in simplex_grid(len(names), steps=steps):
        score = float(average_precision_score(y, stacked @ np.asarray(weights)))
        if score > best_score:
            best_score = score
            best_weights = weights
    assert best_weights is not None
    return {
        "weights": {name: float(w) for name, w in zip(names, best_weights)},
        "average_precision": float(best_score),
        "branch_average_precision": per_branch,
        "status": "fitted",
    }


def apply_fusion(branch_probabilities: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    names = sorted(branch_probabilities)
    missing = [name for name in names if name not in weights]
    if missing:
        raise KeyError(f"missing fusion weights for: {missing}")
    total = sum(float(weights[name]) for name in names)
    if total <= 0:
        raise ValueError("fusion weights must sum to a positive value")
    stacked = np.stack([logit(branch_probabilities[name]) for name in names], axis=1)
    vector = np.asarray([float(weights[name]) / total for name in names])
    return sigmoid(stacked @ vector)
