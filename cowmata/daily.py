"""Day-scale layer: behaviour indicators, individual baselines, oestrus alerts.

Why this is a separate layer and cannot be a tenth event head
-------------------------------------------------------------
Oestrus is a *state* lasting 12-24 h; calving preparation shows in the 24-48 h
before delivery.  One day of 50 Hz nine-axis data is

.. math::

    24 \\times 3600 \\times 50 = 4{,}320{,}000 \\text{ samples}

so no model consumes a day in one forward pass.  "Analyse the waveform
directly" is therefore realised in two stages, not abandoned:

.. math::

    \\underbrace{50\\,\\text{Hz waveform} \\to \\text{minute-scale evidence}}
    _{\\text{the multi-stage model in } \\texttt{models.py}}
    \\;\\to\\;
    \\underbrace{\\text{hour / day aggregation}}_{\\text{this module}}

The second stage is not optional and it is not a downgrade: it is where the
waveform evidence lands on the time scale the biology actually lives on.

The baseline window is the part most projects get wrong
-------------------------------------------------------
Oestrus detection compares an animal against *its own* baseline; between-animal
activity differences are larger than the oestrus effect itself.  Under the
current collection protocol (fit ~3 days before the suspected oestrus, wear for
about 7 days) the days *before* the event are already proestrus - activity has
begun to rise - so a "previous 3 days" baseline is contaminated by the very
signal it is meant to normalise away.

The days *after* the event, metoestrus and early dioestrus, are the cleanest
quiet reference the protocol produces.  :func:`individual_baseline` therefore
supports ``window="after"`` and it is the recommended setting for a 7-day
deployment.  This needs no hardware change and no longer wear time; it is a
change of definition.

Multiple-instance framing
-------------------------
A day is labelled, not a minute.  A cow in oestrus still spends hours lying
still, and those minutes are indistinguishable from a non-oestrus rest.  A
day therefore counts as positive when *enough* of its windows fire, which is the
standard noisy-OR / top-k multiple-instance rule implemented in
:func:`mil_day_score`.  Scoring per window against a day label would penalise
the model for being right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import EVENT_CODES

HOURS_PER_DAY = 24.0

#: Indicators aggregated per cow-day.  Every one is derived from outputs the
#: second-scale model already produces, so route A costs no new training.
DAILY_INDICATORS: tuple[str, ...] = (
    "lying_hours",
    "walking_hours",
    "upright_hours",
    "lying_bouts",
    "posture_transitions",
    "urination_count",
    "defecation_count",
    "tail_raised_count",
    "mounting_count",
    "mounted_by_count",
    "activity_index",
)

#: Direction each indicator is expected to move at oestrus.  Used only to keep
#: the composite score interpretable; nothing here is fitted.
ESTRUS_DIRECTION: dict[str, float] = {
    "lying_hours": -1.0,
    "walking_hours": +1.0,
    "lying_bouts": +1.0,
    "posture_transitions": +1.0,
    "urination_count": +1.0,
    "tail_raised_count": +1.0,
    "mounting_count": +1.0,
    "mounted_by_count": +2.0,  # the veterinary gold standard, weighted double
    "activity_index": +1.0,
}


@dataclass(frozen=True)
class DailyConfig:
    """Aggregation and alerting parameters."""

    baseline_window: str = "after"  # "after" | "before" | "all"
    baseline_min_days: int = 2
    baseline_skip_days: int = 1  # days adjacent to the event that are never baseline
    robust: bool = True  # median / MAD instead of mean / sd
    alert_z: float = 2.5
    mil_top_fraction: float = 0.05
    mil_window_threshold: float = 0.5
    mil_min_windows: int = 3
    #: Robust scales can be exactly zero when a cow's baseline days happen to
    #: be identical (very common for a count that is 0 or 1 every day).  A zero
    #: scale turns any deviation into an infinite z-score, so the spread is
    #: floored relative to the indicator's own level.
    spread_relative_floor: float = 0.10
    spread_absolute_floor: float = 0.05
    #: Deviations are clipped before they are combined.  An unbounded z lets one
    #: degenerate indicator dominate the composite, which is how a single
    #: divide-by-almost-zero silently becomes the whole alert.
    z_clip: float = 10.0


# ==========================================================================
# indicator aggregation
# ==========================================================================
def _count_bouts(state: np.ndarray) -> int:
    values = np.asarray(state).astype(bool)
    if values.size == 0:
        return 0
    return int(np.sum(values[1:] & ~values[:-1]) + (1 if values[0] else 0))


def daily_indicators(
    timeline: pd.DataFrame,
    *,
    cow_column: str = "cow_id",
    day_column: str = "day",
    step_seconds: float = 0.5,
) -> pd.DataFrame:
    """Aggregate a 2 Hz behaviour timeline into one row per cow-day.

    ``timeline`` needs ``cow_id``, ``day``, ``state_lying`` (0/1) and
    ``state_walking`` (0/1), plus one ``event_{CODE}`` column of assembled event
    counts per step, or a set of event interval counts already merged.  Missing
    columns simply produce zero for the indicators that depend on them, so a
    partial pipeline still yields a usable table.
    """

    required = {cow_column, day_column}
    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"timeline is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    hours_per_step = float(step_seconds) / 3600.0
    for (cow, day), group in timeline.groupby([cow_column, day_column], sort=True):
        ordered = group.sort_values("center_time_ms", kind="stable") if "center_time_ms" in group else group
        lying = ordered.get("state_lying")
        walking = ordered.get("state_walking")
        lying_values = (
            np.asarray(lying).astype(bool) if lying is not None else np.zeros(len(ordered), bool)
        )
        walking_values = (
            np.asarray(walking).astype(bool)
            if walking is not None
            else np.zeros(len(ordered), bool)
        )
        record: dict[str, object] = {
            cow_column: cow,
            day_column: day,
            "steps": int(len(ordered)),
            "observed_hours": float(len(ordered) * hours_per_step),
            "lying_hours": float(lying_values.sum() * hours_per_step),
            "walking_hours": float(walking_values.sum() * hours_per_step),
            "upright_hours": float((~lying_values).sum() * hours_per_step),
            "lying_bouts": _count_bouts(lying_values),
            "posture_transitions": int(np.sum(np.diff(lying_values.astype(np.int8)) != 0)),
        }
        for code in EVENT_CODES:
            column = f"event_{code}"
            name = f"{code.lower()}_count"
            record[name] = float(ordered[column].sum()) if column in ordered else 0.0
        record["activity_index"] = float(
            record["walking_hours"] * 2.0 + record["posture_transitions"] * 0.05
        )
        rows.append(record)
    return pd.DataFrame(rows)


# ==========================================================================
# individual baseline
# ==========================================================================
def individual_baseline(
    frame: pd.DataFrame,
    indicator: str,
    *,
    event_day: object = None,
    config: DailyConfig | None = None,
    day_column: str = "day",
) -> tuple[float, float, int]:
    """Return ``(centre, spread, n_days)`` for one cow and one indicator.

    ``event_day`` marks the reference day; days within ``baseline_skip_days`` of
    it are excluded whatever the window setting, because the day either side of
    an oestrus is neither clean baseline nor clean event.
    """

    cfg = config or DailyConfig()
    values = frame[[day_column, indicator]].dropna()
    if event_day is not None:
        days = values[day_column].to_numpy()
        try:
            offset = (days - event_day).astype("timedelta64[D]").astype(int)
        except (TypeError, ValueError):
            offset = np.asarray(days, dtype=float) - float(event_day)
        keep = np.abs(offset) > cfg.baseline_skip_days
        if cfg.baseline_window == "after":
            keep &= offset > 0
        elif cfg.baseline_window == "before":
            keep &= offset < 0
        values = values.loc[keep]

    series = values[indicator].to_numpy(dtype=float)
    series = series[np.isfinite(series)]
    if series.size < cfg.baseline_min_days:
        return float("nan"), float("nan"), int(series.size)
    if cfg.robust:
        centre = float(np.median(series))
        spread = float(np.median(np.abs(series - centre)) * 1.4826)
    else:
        centre = float(np.mean(series))
        spread = float(np.std(series, ddof=1)) if series.size > 1 else 0.0
    floor = max(
        cfg.spread_relative_floor * abs(centre),
        cfg.spread_absolute_floor,
    )
    return centre, max(spread, floor), int(series.size)


def deviation_scores(
    frame: pd.DataFrame,
    *,
    indicators: tuple[str, ...] | None = None,
    cow_column: str = "cow_id",
    day_column: str = "day",
    event_day_column: str | None = None,
    config: DailyConfig | None = None,
) -> pd.DataFrame:
    """Per-cow-day robust z-scores of each indicator against the cow's baseline."""

    cfg = config or DailyConfig()
    chosen = tuple(indicators or [c for c in ESTRUS_DIRECTION if c in frame.columns])
    out = frame.copy()
    for cow, group in frame.groupby(cow_column, sort=True):
        event_day = None
        if event_day_column and event_day_column in group.columns:
            marked = group.loc[group[event_day_column].astype(bool), day_column]
            if len(marked):
                event_day = marked.iloc[0]
        for indicator in chosen:
            centre, spread, n_days = individual_baseline(
                group, indicator, event_day=event_day, config=cfg, day_column=day_column
            )
            index = group.index
            if not np.isfinite(centre):
                out.loc[index, f"z_{indicator}"] = np.nan
            else:
                raw = (group[indicator].to_numpy(dtype=float) - centre) / spread
                out.loc[index, f"z_{indicator}"] = np.clip(raw, -cfg.z_clip, cfg.z_clip)
            out.loc[index, f"baseline_days_{indicator}"] = n_days
    weights = np.asarray([abs(ESTRUS_DIRECTION.get(c, 1.0)) for c in chosen], dtype=float)
    signs = np.asarray([np.sign(ESTRUS_DIRECTION.get(c, 1.0)) for c in chosen], dtype=float)
    matrix = out[[f"z_{c}" for c in chosen]].to_numpy(dtype=float)
    oriented = matrix * signs
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(oriented)
        numerator = np.nansum(np.where(valid, oriented * weights, 0.0), axis=1)
        denominator = np.sum(np.where(valid, weights, 0.0), axis=1)
        out["estrus_score"] = np.where(denominator > 0, numerator / denominator, np.nan)
    out["baseline_window"] = cfg.baseline_window
    return out


def alert_days(frame: pd.DataFrame, *, config: DailyConfig | None = None) -> pd.Series:
    """Boolean alert per cow-day from the composite deviation score."""

    cfg = config or DailyConfig()
    score = frame["estrus_score"].to_numpy(dtype=float)
    return pd.Series(np.isfinite(score) & (score >= cfg.alert_z), index=frame.index)


# ==========================================================================
# multiple-instance aggregation for the end-to-end route
# ==========================================================================
def mil_day_score(
    window_scores: np.ndarray, *, config: DailyConfig | None = None
) -> dict[str, float]:
    """Aggregate minute-scale window scores into one day score.

    Three aggregations are returned because they fail differently and a report
    should show all three: ``top_k_mean`` is robust and is the default decision
    variable, ``noisy_or`` saturates fast and flatters a noisy model, and
    ``fired_windows`` is the count a farm can actually reason about.
    """

    cfg = config or DailyConfig()
    values = np.asarray(window_scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"top_k_mean": float("nan"), "noisy_or": float("nan"), "fired_windows": 0.0, "windows": 0.0}
    k = max(1, int(round(values.size * cfg.mil_top_fraction)))
    top = np.sort(values)[-k:]
    clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
    return {
        "top_k_mean": float(top.mean()),
        "noisy_or": float(1.0 - np.exp(np.sum(np.log1p(-clipped)))),
        "fired_windows": float(np.sum(values >= cfg.mil_window_threshold)),
        "windows": float(values.size),
    }


def mil_day_alert(window_scores: np.ndarray, *, config: DailyConfig | None = None) -> bool:
    cfg = config or DailyConfig()
    summary = mil_day_score(window_scores, config=cfg)
    return bool(
        np.isfinite(summary["top_k_mean"])
        and summary["fired_windows"] >= cfg.mil_min_windows
        and summary["top_k_mean"] >= cfg.mil_window_threshold
    )


def lead_time_hours(alert_time_ms: int, reference_time_ms: int) -> float:
    """Positive when the alert precedes the reference event."""

    return float(reference_time_ms - alert_time_ms) / 3600000.0


def pregnancy_by_return_to_service(
    estrus_days: pd.DataFrame,
    *,
    insemination_day: object,
    cycle_days: int = 21,
    tolerance_days: int = 4,
    day_column: str = "day",
    alert_column: str = "alert",
) -> dict[str, object]:
    """Infer likely pregnancy from the *absence* of a return to oestrus.

    This is deliberately a rule, not a model.  Pregnancy is a months-long
    physiological state with a very weak behavioural signature; there is no
    credible evidence that a tail IMU detects it directly, and a model claiming
    to would be fitting the confound.  What the system *can* do is detect
    oestrus, and a cow that does not return to oestrus about one cycle after
    insemination is the standard on-farm indication to book an ultrasound.  Its
    accuracy is therefore exactly the accuracy of oestrus detection, and it must
    be reported that way rather than as an independent capability.
    """

    days = estrus_days[day_column]
    try:
        offset = (days.to_numpy() - insemination_day).astype("timedelta64[D]").astype(int)
    except (TypeError, ValueError):
        offset = np.asarray(days.to_numpy(), dtype=float) - float(insemination_day)
    window = np.abs(offset - cycle_days) <= tolerance_days
    observed = int(window.sum())
    returned = bool(estrus_days.loc[window, alert_column].astype(bool).any()) if observed else False
    return {
        "insemination_day": insemination_day,
        "cycle_days": int(cycle_days),
        "tolerance_days": int(tolerance_days),
        "days_observed_in_window": observed,
        "returned_to_service": returned,
        "verdict": (
            "not_observed"
            if observed == 0
            else ("open_suspected" if returned else "pregnancy_suspected")
        ),
        "basis": "absence of a detected return to oestrus; confirm by ultrasound",
    }
