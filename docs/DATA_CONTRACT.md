# COWMATA IMU data contract (20260819)

## Raw and cached signal

- Nine-axis IMU sampled continuously at **50 Hz**; the raw JSON is never edited.
- A cached session stores `signal.i16.npy` — `(N, 9) int16` **device counts** — plus
  `meta.json`. Calibration divisors and biases live in the metadata and are applied on
  read, so a calibration correction never requires rebuilding the cache.
- Continuity is defined by `meta.json/segments`. Segments are stored end to end, so no
  window, chunk, filter or rolling statistic may ever cross a segment boundary.
- Within a segment, timestamps are exactly `start_ms + 20 * (index - start_index)`.
- The timing-quality flag is stored as sparse `[lo, hi)` intervals.
- A 20260818 cache (`features.npy`, 13 float32 channels) is read transparently as
  schema 1 through the identical API.

## Required per-session metadata

`cow_id`, `device_mac`, `session_id`, `calibration`, `segments`, and **`tail_position`**
(`root` / `mid` / `tip` / `unknown`). Tail position changes the lever arm and therefore
the angular-rate amplitude of the same behaviour by a large factor; recording a coarse
three-way label costs the field operator one tap and turns an unknown into a covariate.

## Labels

| Tier | Codes | Trained? |
|---|---|---|
| State | `STANDING`, `LYING`, `WALKING`, `FEEDING` | posture (2-way) + walking; `FEEDING` folds into `UPRIGHT` |
| Event | `STANDING_UP`, `LYING_DOWN`, `URINATION`, `DEFECATION`, `TAIL_RAISED`, `MOUNTING`, `MOUNTED_BY` | yes, one sigmoid head each |
| Deprecated | `TAIL_WAGGING` | read without error, never trained, never reported |
| Day scale | `ESTRUS`, `CALVING` | separate layer, see `cowmata/daily.py` |

An event may overlap a state; labels are never collapsed into one mutually exclusive
softmax class.

`tail_raised_policy = derive`: a urination or defecation interval inherits
`TAIL_RAISED = 1`. Under the legacy rule those 104 intervals were confident *negatives*
for the tail-raise head — the model was told that a raised tail during urination is not
a raised tail.

## Splits

- The unit of independence is `cow_id`, never a window and never a session.
- Validation is **cow-disjoint from training**, not merely session-disjoint. Holding out
  sessions of animals that are also in training lets early stopping and threshold
  selection see the identity they are then asked to generalise across.
- Normalisation statistics, early stopping, threshold selection and calibration may
  never inspect a test cow.
- Sample size is reported as independent cows and independent events. Overlapping
  windows are not independent samples.

## Evaluation

See `docs/METRICS.md`. Minimum reportable set: independent cow and event counts, event
precision / recall / F1, `F1@tIoU`, edit score, false alarms per cow per 24 h,
localisation error, and per-cow breakdown with a cow-level bootstrap interval.
