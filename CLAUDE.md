# Claude project context

Read `AGENTS.md`, then `README.md`, `docs/DATA_CONTRACT.md`, `docs/METRICS.md` and
`docs/VERIFICATION_20260819.md`.

The supervised cache is intentionally absent from Git; use `examples/demo_data/` for
executable inspection. Do not infer production metrics from the 60-second demo, from
overlapping windows, or from any pooled cross-cow figure computed on the 20260818 data,
where two of six animals hold ~96% of the supervised samples.

`weights/deploy/gbdt_full.joblib` is the current deployable model and runs at
`feature_version=1`. `weights/checkpoints/offline_tcn_dev_epoch2.pt` is provenance only:
the 20260819 architecture and label set differ, so it cannot be loaded or resumed.
