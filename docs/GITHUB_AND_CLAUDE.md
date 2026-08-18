# GitHub Maintenance and Claude Uploads

## Repository layers

- GitHub: source code, configuration, tests, small annotations/splits, two model artifacts, product/brand assets, and the 60-second demo.
- Local/external data: `supervised_cache/samples.csv` and `supervised_cache/session_cache/`.
- Generated runs: all experiment and prediction outputs under `runs/`.
- Promoted artifacts: only reviewed model weights, manifests, reports, and release notes.

The current model files are approximately 6.7 MB and 8.5 MB, so ordinary Git is sufficient. GitHub warns for files larger than 50 MiB and blocks ordinary Git files larger than 100 MiB. Introduce Git LFS only when a future required artifact crosses that boundary.

## Daily development workflow

```bash
git switch main
git pull --ff-only
git switch -c feature/<short-topic>

# edit, test, and review
pytest
cowmata predict --cache-key demo_session_60s --data-root examples/demo_data --out runs/demo

git add -A
git commit -m "Describe one logical algorithm change"
git push -u origin feature/<short-topic>
```

Use a pull request for substantive algorithm changes. The pull request should record the cows/events used, split policy, leakage checks, metric impact, and test evidence.

## Releases

The repository is `zxq309/cowmata-tailring`.

- `v0.1.0`: clean executable baseline.
- `v0.2.0`: English and visual GitHub documentation release.

Future releases should update `pyproject.toml`, `CHANGELOG.md`, model/data manifests when relevant, tests, and release notes. Do not create another repository for each date.

## Why `dist/` is absent

GitHub already provides clone, archive, tag, and release views. A hand-built ZIP can silently drift from the source commit, so `dist/` is not a source of truth. Generate release artifacts from an exact Git tag only when a consumer actually needs them.

## Claude web uploads

Claude documents currently have a 30 MB per-file limit. Prefer a GitHub integration when the private repository can be authorized. Otherwise upload only source, configuration, tests, README, and text manifests. Do not upload binary weights or the supervised cache.

Temporary source bundles belong under `runs/share/`; they are ignored by Git and should be rebuilt from the current commit rather than manually maintained.

## Credentials

Use OAuth, SSH, or a personal access token. Never store account passwords or tokens in this repository. Rotate any credential that appears in chat, logs, commits, or issue content. The README dataset section records the current snapshot's Baidu Netdisk links and extraction codes by owner decision (20260819); keep future snapshots on the private delivery flow in `docs/DATA_ACCESS.md`.
