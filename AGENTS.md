# COWMATA repository instructions

This repository is the clean `20260818` baseline for the COWMATA smart cattle tail-ring algorithm.

## Non-negotiable data and evaluation rules

- Preserve raw nine-axis IMU continuously at 50 Hz. Windowing is a training parameter, not a raw-data format.
- Align sensor records and video labels on an absolute timeline and retain event start/end timestamps.
- Split train, validation and test data by `cow_id`; never split adjacent windows from one cow across sets.
- Count independent cows and events, not overlapping windows, when reporting sample size.
- Keep the shared temporal encoder, multi-event heads and standing/lying state-machine design compatible.
- Report event Precision/Recall/F1, false alarms per cow per 24 h and localization error; do not use window Accuracy alone.

## Repository boundaries

- Do not commit `datasets/cowmata_imu/supervised_cache/session_cache/`, `samples.csv`, `runs/` or generated archives.
- Keep the 60-second demo session and deploy weights working so a fresh clone can run an inference smoke test.
- New experiments write to timestamped directories under `runs/`; promote only validated weights and reports.
- Preserve compatibility with serialized objects under the `cattle_imu` module unless a migration is supplied.

Before merging algorithm changes, run `pytest` and the demo prediction command documented in `README.md`.
