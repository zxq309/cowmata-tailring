# External Supervised-Cache Delivery and Recovery

## Why the full cache is not stored in GitHub

`supervised_cache` is derived data that is still required by the current training workflow. It is not obsolete material.

- `supervised_cache/samples.csv`: approximately 60 MB and 344,287 supervised center points.
- `supervised_cache/session_cache/`: approximately 1.23 GB, containing 132 continuous 50 Hz sessions.

These artifacts are used to rebuild feature tables, train GBDT/TCN models, run real-data diagnostics, and reproduce cow-level experiments. Ordinary Git is not designed for frequently changing datasets of this size, and the files may contain company, device, and animal identifiers.

GitHub therefore stores only:

- `sessions.csv` and small metadata;
- annotations and cow-level split manifests;
- a 60-second executable demo session;
- model artifacts small enough for ordinary Git.

A clone can run tests and inference without the private cache. Full retraining requires cache recovery.

## Dataset snapshot: 20260818

| Field | Value |
|---|---:|
| Sessions | 132 |
| Continuous duration | 131.5739 hours |
| Supervised center points | 344,287 |
| Local recovery path | `datasets/cowmata_imu/supervised_cache/` |

Required contents:

```text
supervised_cache/
├── samples.csv
├── sessions.csv
└── session_cache/
    └── <cache_key>/
        ├── features.npy
        └── metadata.json
```

After recovery, run:

```bash
cowmata check-data --full-cache-scan
pytest
```

## Recommended delivery process

Baidu Netdisk is acceptable as a temporary private delivery channel. Do not commit a long-lived public link or extraction code to Git.

1. Package the required files as a dated archive such as `cowmata-supervised-cache-20260818.7z`.
2. Compute a SHA-256 hash for the completed archive.
3. Record the archive name, byte size, hash, data date, uploader, and destination project version in a private delivery record.
4. Share the download link and extraction code privately with approved repository members.
5. Verify the SHA-256 before extraction.
6. Extract into the exact recovery path and run the full cache scan.
7. Never overwrite an older archive with a generic name such as `latest.zip`.

As data volume and team size grow, migrate to company-controlled object storage or NAS with account-level access control, immutable versions, retention rules, and audit logs.

## Claude and other web analysis tools

Do not upload the 1.29 GB cache to a web chat. Share source code, configuration, tests, small metadata, and a deliberately de-identified sample instead. Any temporary analysis bundle belongs under `runs/share/`, is never committed, and should be deleted after use.
