# TOCSIN legacy-cache retrospective provenance

The July 2026 TOCSIN cache predates identity-bound cache records. Its static
`tocsin-bart-fastdetectgpt-combined-w-v1` signature is useful evidence, but it
does not prove the resolved BART/Fast-DetectGPT model revisions or the Modal
output lineage. The original `modal-pilot.json` and `modal-full.json` therefore
remain immutable historical records.

Generate two separate, sanitized retrospective manifests from the current
local bytes:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python scripts/generate_tocsin_legacy_provenance.py `
  --run-root outputs/tocsin_retrain_2026-07-10 `
  --dataset-root data `
  --config configs/.profile_full_coverage_with_specdetect.yaml `
  --provenance-out outputs/tocsin_retrain_2026-07-10/manifests/retrospective-cache-provenance.json `
  --cost-out outputs/tocsin_retrain_2026-07-10/manifests/retrospective-cost-projection.json
```

The generator anchors the original Modal manifest hashes, re-runs the current
strict validator over the union and all eight slices, checks every declared
dataset/cache/manifest hash, and corroborates `base_v` against the independently
serialized `fast_detectgpt`/`gpt2` result. It emits no text rows.

`VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND` means the bytes are suitable for
diagnostic evaluation only. `diagnostic_allowed` is true and
`promotion_allowed` is false. Promotion remains locked until a future cache is
fully identity-bound and reports exact status `VERIFIED`, no unbound fields,
and `promotion_allowed: true`.

The corrected cost manifest derives the previously omitted unattributed worker
seconds from the original pilot timings and recomputes the full-run projection
with exact decimal arithmetic. It does not rewrite either original Modal
manifest.
