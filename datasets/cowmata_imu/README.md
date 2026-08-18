# Data and Git Boundary

This directory versions small annotations, session metadata, and cow-level split manifests. The following derived data remains local and is excluded by `.gitignore`:

- `supervised_cache/samples.csv`: 351,128 supervised center points, approximately 59 MB.
- `supervised_cache/session_cache/`: 132 continuous 50 Hz sessions as schema-2
  `signal.i16.npy` (9-channel int16 counts) + `meta.json`, approximately 1.4 GB.
  Schema-1 `features.npy` directories are read transparently.

These artifacts are required to rebuild feature tables, retrain GBDT/MS-TCN++ models,
and run real-data diagnostics. They are not disposable legacy output.

A GitHub clone can complete end-to-end inference smoke testing with `examples/demo_data/`. Formal retraining requires the local cache. See [`../../docs/DATA_ACCESS.md`](../../docs/DATA_ACCESS.md) for external delivery and recovery.

The annotation CSV retains original-language provenance values together with stable English labels and event codes. Algorithm logic should use the code columns; provenance fields should not be translated in place.
