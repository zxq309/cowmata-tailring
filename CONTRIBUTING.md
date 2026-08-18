# Contributing

## Workflow

1. Create a focused branch such as `feature/event-thresholds`.
2. Keep generated data and experiment outputs outside Git.
3. Add or update tests for behavior changes.
4. Run `pytest` and the demo inference command from `README.md`.
5. Open a pull request describing data scope, cow-level split, independent-event counts and metric changes.

Commit messages should describe one logical change. Do not commit passwords, tokens, raw farm data, device identifiers or share-link access codes.

## Releases

The repository is `zxq309/cowmata-tailring`.

- `v0.1.0`: clean executable baseline.
- `v0.2.0`: English and visual GitHub documentation release.
- `v0.3.0`: 20260819 architecture baseline.
- `v0.3.1`: first real-data training run and bilingual experiment report.

Future releases update `pyproject.toml`, `CHANGELOG.md`, model/data manifests when relevant, tests, and release notes. Do not create another repository for each date.

## Why `dist/` is absent

GitHub already provides clone, archive, tag, and release views. A hand-built ZIP can silently drift from the source commit, so `dist/` is not a source of truth. Generate release artifacts from an exact Git tag only when a consumer actually needs them.

## Credentials

Use OAuth, SSH, or a personal access token. Never store account passwords or tokens in this repository. Rotate any credential that appears in chat, logs, commits, or issue content. The README dataset section records the current snapshot's Baidu Netdisk links and extraction codes by owner decision (20260819); keep future snapshots on the private delivery flow in `docs/DATA_ACCESS.md`.

## Access

For the first private-repository phase, repository access is granted by the maintainer. Licensing and public-release terms require company approval.
