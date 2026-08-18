# COWMATA IMU Data Contract

## Source data

- Raw nine-axis IMU is sampled continuously at 50 Hz and retained before any fixed training window is selected.
- Each `features.npy` row contains 13 channels; channel order and units are jointly defined by the cache builder and `metadata.json`.
- Continuous spans are defined by `metadata.json/segments`; windows may not cross a recorded gap.
- Sensor records and video are aligned to an absolute timeline.
- Event annotations retain start and end timestamps.

## Labels

- Persistent states: `STANDING`, `LYING`, and `WALKING`.
- Historical `FEEDING` values may be read but are mapped to auxiliary `UPRIGHT` semantics.
- Posture transitions: `STANDING_UP` and `LYING_DOWN`.
- Elimination and tail events: `URINATION`, `DEFECATION`, `TAIL_RAISED`, and `TAIL_WAGGING`.
- An event may overlap a persistent state; labels must not be collapsed into one mutually exclusive Softmax class.

The Git-tracked annotation table preserves original-language provenance fields together with standardized English labels and codes. Model logic must use the stable code fields.

## Splits and statistics

- Formal generalization evaluation is split by `cow_id`.
- A cow may not appear in both training and test sets.
- Normalization statistics, early stopping, threshold selection, and calibration may not inspect test cows.
- Sample size is reported as independent cows, independent events, and hard negatives.
- Overlapping windows are not independent samples.

## Evaluation

Report at minimum:

- the number of independent cows and events;
- event Precision, Recall, and F1;
- false alarms per cow per 24 hours;
- temporal localization error;
- for calving, first correct alert lead time relative to the delivery anchor.

Window Accuracy may be reported as a diagnostic but is not an acceptance metric.

## Path mapping

The only project data root is `datasets/cowmata_imu/`. Machine-readable paths are centralized in `configs/dataset.yaml`. Large data is excluded from Git, while the local working baseline retains the required cache for reproducibility.
