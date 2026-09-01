# Backing up CAIRN

`origin` is the only copy of this repository's history. This document sets up
a second one.

There are three things worth backing up, and git only covers the first:

| What | Where it lives | Covered by a git mirror? |
| --- | --- | --- |
| Source, configs, tracked model artifacts | git history | yes |
| Runtime data (`/data`) | disk only, gitignored | no — use `--artifacts` |
| Untracked trained profiles (`/models/*`) | disk only, gitignored | no — use `--artifacts` |

`.gitignore` excludes `/data` and all of `/models` except an allowlist of
production profiles. A mirror of the repo is therefore a backup of the
*project*, not of the calibration corpus or the untracked profile grid. Back
those up separately or accept that they are reproducible-but-expensive
(`docs/calibration.md` covers regeneration).

## One-time setup: create the backup repo

This step cannot be automated from a Claude Code session — the GitHub App
token those sessions run under cannot create repositories (`POST /user/repos`
returns 403). Create it by hand:

1. On GitHub, **New repository** → name it `cairn-backup`.
2. Set it **Private**.
3. Do **not** initialise it with a README, `.gitignore`, or licence. The first
   mirror push must land in an empty repo.

## Mirroring

```bash
./scripts/backup_repo.sh --remote https://github.com/<you>/cairn-backup.git
```

This pushes every branch and tag that exists on `origin` to the backup remote,
pruning refs there that origin no longer has — so the backup tracks deletions
instead of accumulating dead branches.

Two things it deliberately does *not* do:

- **It mirrors `origin`, not your working clone.** Local branches you have not
  pushed are not backed up. Push first, then mirror.
- **It refuses to back up a truncated history silently.** If the clone is
  shallow (CI checkouts and Claude Code containers usually are) the script
  runs `git fetch --unshallow` first. A shallow mirror is the dangerous kind
  of backup: it looks complete.

After the first push, set the backup repo's default branch to `main` in its
GitHub settings — GitHub picks one arbitrarily when several branches arrive in
the same push.

### Automated mirroring

`.github/workflows/backup-mirror.yml` runs the same push on every commit to
`main`, weekly, and on demand. It is inert until two repository secrets exist:

- `BACKUP_REMOTE_URL` — e.g. `https://github.com/<you>/cairn-backup.git`
- `BACKUP_TOKEN` — a PAT with `contents: write` on the backup repo only

Without them the job logs that it is unconfigured and exits 0, so the workflow
never fails CI for a repo that has not opted in.

## Offline snapshots

A `git bundle` is a single file that `git clone` accepts as a source. Good for
cold storage, an external drive, or anywhere a second remote is not an option:

```bash
./scripts/backup_repo.sh --bundle ./backups --artifacts ./backups
```

`--artifacts` adds a tarball of the gitignored files under `data/` and
`models/` — precisely the files `git ls-files --others --ignored` reports, so
`.gitignore` stays the single source of truth for what counts as an artifact.
Pass `--artifact-paths "data models outputs"` to widen it.

Bundles are verified with `git bundle verify` immediately after being written;
the script fails if verification does not pass.

## Restoring

From the backup remote:

```bash
git clone https://github.com/<you>/cairn-backup.git cairn
```

From a bundle:

```bash
git clone --branch main cairn-<timestamp>.bundle cairn
cd cairn
git remote set-url origin https://github.com/<you>/cairn.git
```

Then unpack any artifact tarball over the restored tree:

```bash
tar xzf cairn-artifacts-<timestamp>.tar.gz -C cairn/
```

Verify the restore the same way any other checkout is verified:

```bash
pip install -e ".[dev]"
pytest -q
```

## Checking a backup is real

A backup nobody has restored is a hypothesis. Periodically:

```bash
git clone --bare https://github.com/<you>/cairn-backup.git /tmp/verify.git
git --git-dir=/tmp/verify.git fsck
git --git-dir=/tmp/verify.git log --oneline -1 main   # matches origin/main?
```
