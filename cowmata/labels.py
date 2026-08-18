"""Single source of truth for the COWMATA label taxonomy.

Design notes
------------
The 20260818 baseline scattered code lists over ``annotations.py``,
``preprocessing.py``, ``dataset.py`` and three training scripts.  Every list had
to agree or the cache silently mislabelled itself.  They now live here and
nowhere else.

Three tiers exist and they are *not* interchangeable:

``STATE_ANNOTATION_CODES``
    Mutually exclusive body states the annotator marks.  ``FEEDING`` is still
    read so historical work is never discarded, but a tail ring cannot separate
    feeding from plain standing, so it is folded into ``UPRIGHT`` and is not a
    prediction target.

``EVENT_CODES``
    Second-scale discrete events, the trained multi-label outputs.
    ``MOUNTING`` / ``MOUNTED_BY`` are new in 20260819: being mounted is the
    veterinary gold standard for oestrus and it is exactly the kind of short
    impact event a tail-root IMU sees well.

``DEPRECATED_EVENT_CODES``
    Codes that remain readable in the annotation table but are excluded from
    training and from every reported metric.  ``TAIL_WAGGING`` has 4 intervals
    from 1 animal; keeping it in the loss let one animal's 4 intervals carry the
    same weight as 50,000 posture labels in the old equally-averaged selection
    score.

``DAY_LEVEL_CODES``
    Physiological states annotated at day scale from breeding records,
    ultrasound and veterinary follow-up.  They are never mixed into the
    second-scale event heads; see :mod:`cowmata.daily`.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# body state
# --------------------------------------------------------------------------
STATE_ANNOTATION_CODES: tuple[str, ...] = ("STANDING", "LYING", "WALKING", "FEEDING")

#: Backward-compatible alias.  The 20260818 supervised cache stores
#: ``body_target`` as an index into this exact tuple, so the order is frozen.
BODY_CODES = STATE_ANNOTATION_CODES

POSTURE_CODES: tuple[str, ...] = ("UPRIGHT", "LYING")
LOCOMOTION_CODE = "WALKING"

#: STATE_ANNOTATION_CODES index -> POSTURE_CODES index.
STATE_TO_POSTURE: dict[int, int] = {
    0: 0,  # STANDING -> UPRIGHT
    1: 1,  # LYING    -> LYING
    2: 0,  # WALKING  -> UPRIGHT
    3: 0,  # FEEDING  -> UPRIGHT
}

# --------------------------------------------------------------------------
# second-scale events
# --------------------------------------------------------------------------
EVENT_CODES: tuple[str, ...] = (
    "STANDING_UP",
    "LYING_DOWN",
    "URINATION",
    "DEFECATION",
    "TAIL_RAISED",
    "MOUNTING",
    "MOUNTED_BY",
)

DEPRECATED_EVENT_CODES: tuple[str, ...] = ("TAIL_WAGGING",)

#: Every code the annotation loader accepts without raising.
ALL_ANNOTATION_CODES: tuple[str, ...] = (
    STATE_ANNOTATION_CODES + EVENT_CODES + DEPRECATED_EVENT_CODES
)

# --------------------------------------------------------------------------
# day-scale physiological states
# --------------------------------------------------------------------------
DAY_LEVEL_CODES: tuple[str, ...] = ("ESTRUS", "CALVING")

EVENT_LABELS_ZH: dict[str, str] = {
    "STANDING_UP": "起立",
    "LYING_DOWN": "卧倒",
    "URINATION": "排尿",
    "DEFECATION": "排便",
    "TAIL_RAISED": "抬尾",
    "MOUNTING": "爬跨",
    "MOUNTED_BY": "被爬跨",
    "TAIL_WAGGING": "甩尾",
}

EVENT_LABELS_EN: dict[str, str] = {
    "STANDING_UP": "standing up",
    "LYING_DOWN": "lying down",
    "URINATION": "urination",
    "DEFECATION": "defecation",
    "TAIL_RAISED": "tail raised",
    "MOUNTING": "mounting",
    "MOUNTED_BY": "mounted by another cow",
    "TAIL_WAGGING": "tail wagging",
}

STATE_LABELS_ZH: dict[str, str] = {
    "STANDING": "站立",
    "LYING": "躺卧",
    "WALKING": "行走",
    "FEEDING": "采食",
    "UPRIGHT": "直立",
}

DAY_LABELS_ZH: dict[str, str] = {"ESTRUS": "发情", "CALVING": "产犊"}

# --------------------------------------------------------------------------
# per-event physical priors
# --------------------------------------------------------------------------
#: Published per-animal rates for housed dairy cattle, events per hour.  Used
#: only as a sanity ceiling in reporting and in candidate mining, never as a
#: training target and never inside threshold selection (doing that on a
#: label-enriched subset silently destroys recall).
#:
#: urination  ~9/24 h  -> 0.38/h, generous bound 1.5/h
#: defecation ~16/24 h -> 0.67/h, generous bound 2.5/h
#: mounting activity concentrates in oestrus, so its bound is a daily peak
#: rather than a herd average.
PHYSIOLOGICAL_RATE_PER_HOUR: dict[str, float] = {
    "URINATION": 1.5,
    "DEFECATION": 2.5,
    "STANDING_UP": 4.0,
    "LYING_DOWN": 4.0,
    "TAIL_RAISED": 12.0,
    "MOUNTING": 8.0,
    "MOUNTED_BY": 8.0,
    "TAIL_WAGGING": 60.0,
}

#: Interval post-processing.  ``merge_gap_ms`` closes a probability dip inside
#: one real event; ``min_ms`` drops fragments too short to be that behaviour.
#: Both are deliberately shorter than the observed annotation durations so a
#: real event is never dissolved.
EVENT_POSTPROCESS: dict[str, dict[str, int]] = {
    "STANDING_UP": {"merge_gap_ms": 2000, "min_ms": 1000},
    "LYING_DOWN": {"merge_gap_ms": 2000, "min_ms": 1000},
    "URINATION": {"merge_gap_ms": 5000, "min_ms": 4000},
    "DEFECATION": {"merge_gap_ms": 4000, "min_ms": 3000},
    "TAIL_RAISED": {"merge_gap_ms": 3000, "min_ms": 2000},
    "MOUNTING": {"merge_gap_ms": 2000, "min_ms": 1000},
    "MOUNTED_BY": {"merge_gap_ms": 2000, "min_ms": 1000},
    "TAIL_WAGGING": {"merge_gap_ms": 1500, "min_ms": 1000},
}

#: Temporal context each event head needs, in seconds.  Defecation is the
#: longest observed behaviour; mounting is an impact and needs almost none.
EVENT_CONTEXT_SECONDS: dict[str, float] = {
    "STANDING_UP": 8.0,
    "LYING_DOWN": 8.0,
    "URINATION": 20.0,
    "DEFECATION": 30.0,
    "TAIL_RAISED": 15.0,
    "MOUNTING": 6.0,
    "MOUNTED_BY": 6.0,
    "TAIL_WAGGING": 3.0,
}

# --------------------------------------------------------------------------
# evidence gating
# --------------------------------------------------------------------------
#: Minimum evidence before precision / F1 may be reported for an event at all.
MIN_TRUE_EVENTS_FOR_CLAIM = 10
MIN_COWS_FOR_CLAIM = 3


def trained_event_codes() -> tuple[str, ...]:
    """Events the model actually learns and reports."""

    return EVENT_CODES


def is_deprecated(code: str) -> bool:
    return str(code).upper() in DEPRECATED_EVENT_CODES


def label_zh(code: str) -> str:
    key = str(code).upper()
    return (
        EVENT_LABELS_ZH.get(key)
        or STATE_LABELS_ZH.get(key)
        or DAY_LABELS_ZH.get(key, key)
    )


def label_en(code: str) -> str:
    key = str(code).upper()
    return EVENT_LABELS_EN.get(key, key.lower().replace("_", " "))
