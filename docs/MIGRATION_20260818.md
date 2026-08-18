# 20260818 Migration Record

## Retained

- The `cattle_imu` algorithm core and original contract tests.
- The package path required by the serialized `cattle_imu.gbdt.BinaryBooster` artifact.
- The operational GBDT model and its feature-extraction/candidate-merging path.
- Required annotations, cow-level splits, and supervised-cache interface.
- One incomplete offline TCN development checkpoint for continuation and engineering smoke tests only.

## Removed or excluded

- Python bytecode, Pytest caches, old logs, one-off scan scripts, and temporary backups.
- Duplicate predictions, intermediate feature tables, and failed/incomplete run directories.
- Entry points bound to personal disk paths or the earlier `20260816` absolute layout.
- SSL/VICReg cache, window index, pretraining, and prototype-retrieval branches not used by the retained GBDT or TCN artifacts.
- A Chinese binary design document from the GitHub deployment; the local source copy remains available outside version control.

## Structural changes

- Consolidated project data under `datasets/cowmata_imu/`.
- Consolidated generated outputs under `runs/`.
- Consolidated model artifacts under `weights/`.
- Added the model-first `from cowmata import COWMATA` API and unified `cowmata` CLI.
- Retained `scripts/predict_full.py` as a compatibility entry point routed to the unified CLI.
- Converted split-manifest paths to repository-relative paths without changing array values, timestamps, labels, or split membership.
- Added a 60-second real-session demo while excluding the large supervised cache from Git.
- Added English GitHub documentation, official product assets, CI, governance, and reference-project tracking.

No earlier dated iteration directory was deleted or overwritten. “Removed” means excluded from the clean `20260818` baseline.
