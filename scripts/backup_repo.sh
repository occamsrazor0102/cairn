#!/usr/bin/env bash
# Mirror this repository to a backup destination.
#
# Two independent backup targets, usable together:
#   --remote <url>    push every origin branch + tag to a second git remote
#                     (the "backup repo"). Refs on the destination that no
#                     longer exist on origin are pruned, so the backup is a
#                     true mirror rather than an append-only pile.
#   --bundle <dir>    write a single-file `git bundle` snapshot into <dir>.
#                     A bundle is a self-contained clone source — good for
#                     offline / cold storage where a second remote isn't one.
#
# Git only carries tracked files, and this repo gitignores its heavy runtime
# state (/data, most of /models — see .gitignore). Neither target above backs
# those up. Add --artifacts to also write a tarball of the ignored-but-present
# files under those paths.
#
# Prereqs:
#   - push access to the destination remote (a PAT in the URL, a credential
#     helper, or SSH). This script never handles credentials itself.
#   - the destination repo must already exist and should be EMPTY on first
#     run — an unrelated repo would have its refs pruned away.
#
# Usage:
#   ./scripts/backup_repo.sh --remote https://github.com/<you>/cairn-backup.git
#   ./scripts/backup_repo.sh --bundle ./backups --artifacts ./backups
#   ./scripts/backup_repo.sh --remote <url> --dry-run
#
# Options:
#   --remote <url>          mirror-push to this git remote
#   --bundle <dir>          write a .bundle snapshot into this directory
#   --artifacts <dir>       write a .tar.gz of gitignored artifacts into <dir>
#   --artifact-paths "a b"  paths the artifact tarball covers (default: "data models")
#   --dry-run               show what would be pushed/written, change nothing
#   -h, --help              this help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

remote=""
bundle_dir=""
artifacts_dir=""
artifact_paths="data models"
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)         remote="${2:?--remote needs a url}"; shift 2 ;;
        --bundle)         bundle_dir="${2:?--bundle needs a directory}"; shift 2 ;;
        --artifacts)      artifacts_dir="${2:?--artifacts needs a directory}"; shift 2 ;;
        --artifact-paths) artifact_paths="${2:?--artifact-paths needs a path list}"; shift 2 ;;
        --dry-run)        dry_run=1; shift ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$remote" && -z "$bundle_dir" && -z "$artifacts_dir" ]]; then
    echo "nothing to do: pass at least one of --remote, --bundle, --artifacts" >&2
    echo "see: $0 --help" >&2
    exit 2
fi

# The mirror push prunes. Aimed at origin by mistake it would delete branches
# on the very repo being backed up, so refuse that before touching anything.
if [[ -n "$remote" ]]; then
    origin_url="$(git remote get-url origin 2>/dev/null || true)"
    if [[ -n "$origin_url" && "$remote" == "$origin_url" ]]; then
        echo "refusing to mirror onto origin ($origin_url):" >&2
        echo "  --remote must point at a separate backup repository." >&2
        exit 1
    fi
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

# ---------------------------------------------------------------- full history
# A shallow clone would silently produce a truncated backup, which is worse
# than no backup: it looks complete. Deepen before mirroring anything.
if [[ "$(git rev-parse --is-shallow-repository)" == "true" ]]; then
    echo "==> clone is shallow; fetching full history before backup"
    if [[ "$dry_run" -eq 1 ]]; then
        echo "    (dry-run) would run: git fetch --unshallow --tags origin"
    else
        git fetch --unshallow --tags origin
    fi
fi

if [[ -n "$remote" || -n "$bundle_dir" ]]; then
    echo "==> refreshing origin refs"
    if [[ "$dry_run" -eq 1 ]]; then
        echo "    (dry-run) would run: git fetch --prune --tags origin"
    else
        git fetch --prune --tags origin
    fi
fi

# ------------------------------------------------------------------- mirror push
if [[ -n "$remote" ]]; then
    # Map origin's branches onto the backup's branches. Pushing --mirror from a
    # working clone would copy refs/remotes/* verbatim and leave the backup with
    # nested origin/* refs, so build explicit refspecs instead. origin/HEAD is a
    # symref, not a branch — excluded.
    refspecs=()
    while IFS= read -r line; do
        [[ "$line" == *":refs/heads/HEAD" ]] && continue
        refspecs+=("$line")
    done < <(git for-each-ref \
        --format='+%(refname):refs/heads/%(refname:lstrip=3)' refs/remotes/origin)

    if [[ ${#refspecs[@]} -eq 0 ]]; then
        echo "no origin branches found to mirror" >&2
        exit 1
    fi
    refspecs+=("+refs/tags/*:refs/tags/*")

    echo "==> mirroring ${#refspecs[@]} refspec(s) to $remote"
    for spec in "${refspecs[@]}"; do
        echo "    $spec"
    done

    if [[ "$dry_run" -eq 1 ]]; then
        git push --prune --dry-run "$remote" "${refspecs[@]}"
    else
        git push --prune "$remote" "${refspecs[@]}"
        echo "==> mirror push complete"
    fi
fi

# ----------------------------------------------------------------- bundle file
if [[ -n "$bundle_dir" ]]; then
    bundle_path="$bundle_dir/cairn-$timestamp.bundle"
    echo "==> writing bundle: $bundle_path"
    if [[ "$dry_run" -eq 1 ]]; then
        echo "    (dry-run) would run: git bundle create <path> --all"
    else
        mkdir -p "$bundle_dir"
        git bundle create "$bundle_path" --all
        # A bundle that can't be verified is not a backup. Fail loudly.
        git bundle verify "$bundle_path" >/dev/null
        echo "==> bundle verified ($(du -h "$bundle_path" | cut -f1))"
    fi
fi

# ------------------------------------------------------- gitignored artifacts
if [[ -n "$artifacts_dir" ]]; then
    artifact_path="$artifacts_dir/cairn-artifacts-$timestamp.tar.gz"
    list_file="$(mktemp)"
    trap 'rm -f "$list_file"' EXIT

    # Exactly the files git would not carry: present on disk, ignored, under
    # the requested paths. Nothing is hardcoded, so .gitignore stays the single
    # source of truth for what counts as an artifact.
    # shellcheck disable=SC2086
    git ls-files --others --ignored --exclude-standard -z -- $artifact_paths \
        > "$list_file" || true

    if [[ ! -s "$list_file" ]]; then
        echo "==> no gitignored artifacts under: $artifact_paths (nothing to archive)"
    else
        count="$(tr -cd '\0' < "$list_file" | wc -c)"
        echo "==> archiving $count gitignored file(s) -> $artifact_path"
        if [[ "$dry_run" -eq 1 ]]; then
            echo "    (dry-run) would run: tar czf <path> --null -T <filelist>"
        else
            mkdir -p "$artifacts_dir"
            tar czf "$artifact_path" --null -T "$list_file"
            echo "==> artifacts written ($(du -h "$artifact_path" | cut -f1))"
        fi
    fi
fi

echo "==> done"
