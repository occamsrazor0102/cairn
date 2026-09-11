"""Screen a JSONL of texts through every available calibrated profile with a
phase-restricted cascade, and write a profile x sample results matrix.

Question this answers: "what does each calibrated profile say about these
samples when only the requested cascade phases run?" -- without paying the
heavy detector cost once per profile.

Detector outputs are calibrator-INDEPENDENT (see cairn/feature_cache.py), so
the script:

  1. discovers every profile under configs/ (dot-prefixed research profiles
     included) and checks whether its fusion artifacts are all present on
     disk -- only those are run; the rest are listed with the reason;
  2. runs the raw detector pass ONCE with an uncalibrated superset config
     (default.yaml + every requested-phase detector any available profile
     enables), forced through exactly the requested phases, and caches it
     with cairn.feature_cache.write_record;
  3. replays that cache through each profile's detector calibrator + EBM +
     conformal wrapper + decision engine via CachedPipeline. A profile whose
     detector config differs from the superset (different scoring model,
     router thresholds, ...) is run directly instead, so replay is only used
     where it is provably equivalent;
  4. optionally re-runs one profile directly end-to-end (--verify-direct) and
     reports the max |p_llm| difference against the replay;
  5. writes results.json, results_long.csv, detectors_long.csv,
     profiles.json and (with --xlsx) a results workbook.

Phase restriction (--phases, default "1,2"): cascade thresholds are set so
every requested phase fires on every sample and later phases never fire;
detectors belonging to excluded phases are also disabled explicitly. Phase 4
is batch-only and never part of a single-submission screen.

Also recorded per profile x sample: what the profile's OWN cascade
thresholds would have done with the phase-1 evidence (would it have
escalated to phase 2 / phase 3?), so the forced run can be read against the
production short-circuit behaviour.

Usage (from the repo root, with cairn[fusion,tier2] installed):

    python scripts/run_profile_screen.py \\
        --dataset data/buckeye_test_prompts.jsonl \\
        --phases 1,2 \\
        --out-dir docs/benchmarks/buckeye_prompts \\
        --verify-direct master_minus_arxiv_robust_v1 \\
        --xlsx

Dataset rows need "id" and "text"; any other keys are carried through as
sample metadata (labels are NOT required -- this is a screen, not a
benchmark).
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from cairn.cascade import (
    PHASE1_DETECTORS,
    PHASE2_DETECTORS,
    PHASE3_DETECTORS,
    PHASE4_DETECTORS,
    CascadeScheduler,
)
from cairn.config import CONFIGS_DIR, load_config
from cairn.contracts import Determination, LayerResult, RouterDecision
from cairn.feature_cache import CachedPipeline, load_cache, text_hash, write_record
from cairn.pipeline import CairnPipeline, detector_availability, normalize_text

ROOT = Path(__file__).resolve().parent.parent

PHASE_DETECTORS: dict[int, tuple[str, ...]] = {
    1: PHASE1_DETECTORS,
    2: PHASE2_DETECTORS,
    3: PHASE3_DETECTORS,
    4: PHASE4_DETECTORS,
}
PHASE_OF: dict[str, int] = {
    det_id: phase for phase, ids in PHASE_DETECTORS.items() for det_id in ids
}
# Config keys that do not change what a detector computes (only how fusion
# weighs it / whether it runs), so they are ignored when deciding whether a
# profile can replay the shared raw cache.
_NON_COMPUTE_KEYS = frozenset({"enabled", "weight"})

_SKOPS_MAGIC = b"PK\x03\x04"


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

@dataclass
class ProfileInfo:
    name: str                       # display name (file stem minus ".profile_")
    declared_profile: str | None    # `_profile` key from the YAML
    path: str                       # repo-relative config path
    extends: str | None
    model_path: str | None
    conformal_path: str | None
    detector_calibration_path: str | None
    status: str                     # available | unavailable | not-calibrated | unparseable
    missing: list[str] = field(default_factory=list)
    note: str = ""
    config: dict | None = field(default=None, repr=False)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("config", None)
        return d


def _display_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith(".profile_"):
        return stem[len(".profile_"):]
    return stem.lstrip(".")


def _artifact_problems(model_path: str | None, conformal_path: str | None,
                       detector_calibration_path: str | None) -> list[str]:
    problems: list[str] = []
    for label, rel in (
        ("model", model_path),
        ("conformal", conformal_path),
        ("detector_calibration", detector_calibration_path),
    ):
        if not rel:
            problems.append(f"{label}: not configured")
        elif not (ROOT / rel).exists():
            problems.append(f"{label}: {rel} not found")
    if model_path and (ROOT / model_path).exists():
        with open(ROOT / model_path, "rb") as fh:
            head = fh.read(4)
        if head == _SKOPS_MAGIC:
            sidecar = ROOT / f"{model_path}.hmac"
            if not sidecar.exists():
                problems.append(
                    f"model: HMAC sidecar {model_path}.hmac not found "
                    "(cairn.fusion.serialization.load_model refuses skops "
                    "artifacts without one)"
                )
        elif not os.environ.get("CAIRN_ALLOW_LEGACY_PICKLE"):
            problems.append(
                f"model: {model_path} is a legacy pickle "
                "(set CAIRN_ALLOW_LEGACY_PICKLE=1 to opt in)"
            )
    return problems


def discover_profiles(configs_dir: Path = CONFIGS_DIR) -> list[ProfileInfo]:
    """Every configs/*.yaml and configs/.*.yaml, classified by artifact state."""
    # pathlib's "*" also matches dot-prefixed files, so this covers the
    # research profiles (configs/.profile_*.yaml) as well as the public ones.
    paths = sorted(configs_dir.glob("*.yaml"))
    infos: list[ProfileInfo] = []
    for p in paths:
        if p.name.startswith("_"):
            continue  # helper configs (e.g. _cache_builder.yaml)
        rel = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cfg = load_config(p)
        except Exception as exc:  # noqa: BLE001 - report, don't abort discovery
            infos.append(ProfileInfo(
                name=_display_name(p), declared_profile=None, path=rel,
                extends=None, model_path=None, conformal_path=None,
                detector_calibration_path=None, status="unparseable",
                note=f"{type(exc).__name__}: {exc}",
            ))
            continue
        fusion = cfg.get("fusion") or {}
        info = ProfileInfo(
            name=_display_name(p),
            declared_profile=raw.get("_profile"),
            path=rel,
            extends=raw.get("_extends"),
            model_path=fusion.get("model_path"),
            conformal_path=fusion.get("conformal_path"),
            detector_calibration_path=fusion.get("detector_calibration_path"),
            status="not-calibrated",
            config=cfg,
        )
        if not info.model_path:
            info.note = "no fusion.model_path (heuristic profile)"
        else:
            info.missing = _artifact_problems(
                info.model_path, info.conformal_path, info.detector_calibration_path,
            )
            info.status = "available" if not info.missing else "unavailable"
        infos.append(info)
    return infos


# --------------------------------------------------------------------------
# Phase restriction
# --------------------------------------------------------------------------

def parse_phases(spec: str) -> tuple[int, ...]:
    try:
        phases = tuple(sorted({int(x) for x in spec.split(",") if x.strip()}))
    except ValueError as exc:
        raise SystemExit(f"--phases: {exc}") from exc
    if phases not in ((1,), (1, 2), (1, 2, 3)):
        raise SystemExit("--phases must be one of: 1 | 1,2 | 1,2,3 (phase 4 is batch-only)")
    return phases


def apply_phase_overlay(cfg: dict, phases: tuple[int, ...]) -> dict:
    """Return a copy of cfg whose cascade fires exactly `phases` on every text.

    `CascadeScheduler.should_run_phase2/3` test `low < agg < high` with
    agg in [0, 1], so bounds strictly outside that interval force the check
    open (-0.01 / 1.01) or closed (1.01 / -0.01). Detectors of excluded
    phases are disabled as well so the cascade thresholds are not the only
    thing keeping them out.
    """
    cfg = copy.deepcopy(cfg)
    cascade = cfg.setdefault("cascade", {})
    open_, closed = (-0.01, 1.01), (1.01, -0.01)
    p2_low, p2_high = open_ if 2 in phases else closed
    p3_low, p3_high = open_ if 3 in phases else closed
    cascade["phase1_short_circuit_low"] = p2_low
    cascade["phase1_short_circuit_high"] = p2_high
    cascade["phase2_short_circuit_low"] = p3_low
    cascade["phase2_short_circuit_high"] = p3_high
    cascade["phase3_always_run"] = 3 in phases
    detectors = cfg.setdefault("detectors", {})
    for phase, ids in PHASE_DETECTORS.items():
        if phase in phases:
            continue
        for det_id in ids:
            block = dict(detectors.get(det_id) or {})
            block["enabled"] = False
            detectors[det_id] = block
    # A screen must not touch the side-channel stores under data/.
    for key in ("monitoring", "audit", "retrieval"):
        block = dict(cfg.get(key) or {})
        block["enabled"] = False
        cfg[key] = block
    return cfg


def build_superset_config(profiles: list[ProfileInfo], phases: tuple[int, ...]) -> dict:
    """default.yaml, uncalibrated, plus every requested-phase detector that any
    available profile enables (copying that profile's detector block)."""
    base = load_config(CONFIGS_DIR / "default.yaml")
    base["_profile"] = "profile_screen_raw_superset"
    base["fusion"] = {
        "model_path": None, "conformal_path": None, "detector_calibration_path": None,
    }
    detectors = base.setdefault("detectors", {})
    for phase in phases:
        for det_id in PHASE_DETECTORS[phase]:
            if (detectors.get(det_id) or {}).get("enabled", True):
                continue
            for prof in profiles:
                block = ((prof.config or {}).get("detectors") or {}).get(det_id) or {}
                if block.get("enabled", True):
                    detectors[det_id] = copy.deepcopy(block)
                    break
    return apply_phase_overlay(base, phases)


def _compute_block(block: dict | None) -> dict:
    return {k: v for k, v in (block or {}).items() if k not in _NON_COMPUTE_KEYS}


def replay_incompatibilities(profile_cfg: dict, superset_cfg: dict,
                             phases: tuple[int, ...]) -> list[str]:
    """Detector ids (or 'router') whose compute-relevant config differs from
    the superset that produced the shared cache. Empty list = exact replay."""
    diffs: list[str] = []
    prof_dets = profile_cfg.get("detectors") or {}
    sup_dets = superset_cfg.get("detectors") or {}
    for phase in phases:
        for det_id in PHASE_DETECTORS[phase]:
            block = prof_dets.get(det_id) or {}
            if not block.get("enabled", True):
                continue
            if _compute_block(block) != _compute_block(sup_dets.get(det_id)):
                diffs.append(det_id)
    if (profile_cfg.get("router") or {}) != (superset_cfg.get("router") or {}):
        diffs.append("router")
    return diffs


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def load_samples(path: Path, limit: int | None) -> list[dict]:
    samples: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "text"):
                if key not in row:
                    raise SystemExit(f"{path}: line {lineno} missing '{key}'")
            samples.append(row)
    if limit is not None:
        samples = samples[:limit]
    return samples


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample_meta(sample: dict) -> dict:
    meta = {k: v for k, v in sample.items() if k not in ("text", "id")}
    meta["sample_id"] = sample["id"]
    return meta


# --------------------------------------------------------------------------
# Raw (uncalibrated) detector pass -> feature cache
# --------------------------------------------------------------------------

def _cache_signature(superset_cfg: dict) -> str:
    payload = json.dumps(
        {"detectors": superset_cfg.get("detectors"), "cascade": superset_cfg.get("cascade"),
         "router": superset_cfg.get("router")},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_raw_cache(samples: list[dict], superset_cfg: dict, cache_path: Path,
                    *, rebuild: bool) -> dict:
    """Run the superset pipeline on every sample not already cached."""
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    signature = _cache_signature(superset_cfg)
    existing: dict[str, dict] = {}
    if cache_path.exists() and not rebuild:
        stale = True
        if meta_path.exists():
            try:
                stale = json.loads(meta_path.read_text()).get("signature") != signature
            except (OSError, json.JSONDecodeError):
                stale = True
        if stale:
            print(f"[cache] {cache_path} was built with a different superset config; rebuilding", flush=True)
        else:
            existing = load_cache(cache_path)
            print(f"[cache] reusing {len(existing)} cached records from {cache_path}", flush=True)
    pipeline = CairnPipeline(superset_cfg)
    if pipeline._detector_calibrator is not None:  # pragma: no cover - defensive
        raise RuntimeError("raw pass must run WITHOUT a detector calibrator")
    availability = detector_availability(superset_cfg)
    unavailable = [row for row in availability if row["enabled"] and not row["available"]]
    for row in unavailable:
        print(f"[raw] detector {row['id']} enabled but unavailable: {row['reason']}", flush=True)
    info = {
        "cache_path": str(cache_path),
        "signature": signature,
        "detector_availability": availability,
        "detector_errors": {},
        "seconds_per_sample": {},
        "n_computed": 0,
        "n_reused": 0,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing else "w"
    with open(cache_path, mode, encoding="utf-8") as out:
        for i, sample in enumerate(samples, start=1):
            text = normalize_text(sample["text"])
            key = text_hash(text)
            if key in existing:
                info["n_reused"] += 1
                continue
            t0 = time.monotonic()
            det, results, rd = pipeline.analyze_detailed(text)
            elapsed = round(time.monotonic() - t0, 2)
            info["seconds_per_sample"][sample["id"]] = elapsed
            if det.detector_errors:
                info["detector_errors"][sample["id"]] = det.detector_errors
            write_record(out, key, results, rd)
            out.flush()
            existing[key] = {"results": [], "router": {}}
            info["n_computed"] += 1
            fired = [r.layer_id for r in results if r.determination != "SKIP"]
            print(
                f"[raw] {i}/{len(samples)} {sample['id']}: router={rd.domain} "
                f"words={rd.word_count} phases={det.cascade_phases} "
                f"fired={len(fired)} skipped={len(results) - len(fired)} "
                f"errors={len(det.detector_errors)} ({elapsed}s)",
                flush=True,
            )
    meta_path.write_text(json.dumps({"signature": signature,
                                     "built_at": _now_iso()}, indent=2))
    return info


# --------------------------------------------------------------------------
# Per-profile evaluation
# --------------------------------------------------------------------------

def _finite(x) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _scalar_signals(signals: dict) -> dict:
    out = {}
    for k, v in (signals or {}).items():
        if hasattr(v, "item") and not isinstance(v, (str, bytes, list, dict)):
            try:
                v = v.item()  # numpy scalar -> python scalar
            except (TypeError, ValueError):
                pass
        if isinstance(v, bool) or v is None:
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = _finite(v)
        elif isinstance(v, str):
            out[k] = v if len(v) <= 120 else v[:117] + "..."
    return out


def detector_row(result: LayerResult) -> dict:
    return {
        "layer_id": result.layer_id,
        "phase": PHASE_OF.get(result.layer_id),
        "domain": result.domain,
        "raw_score": _finite(result.raw_score),
        "p_llm": _finite(result.p_llm),
        "confidence": _finite(result.confidence),
        "determination": result.determination,
        "calibrator_applied": bool(result.calibrator_applied),
        "calibrator_flags": sorted(k for k in (result.signals or {}) if k.startswith("calibrator_")),
        "skipped_reason": (result.signals or {}).get("skipped"),
        "signals": _scalar_signals(result.signals),
    }


def summarize(prof: ProfileInfo, sample: dict, det: Determination,
              results: list[LayerResult], rd: RouterDecision,
              own_cascade: CascadeScheduler) -> dict:
    phase1 = [r for r in results if r.layer_id in PHASE1_DETECTORS]
    would_p2 = own_cascade.should_run_phase2(phase1)
    would_p3 = would_p2 and own_cascade.should_run_phase3(results)
    row = {
        "profile": prof.name,
        **_sample_meta(sample),
        "router_domain": rd.domain,
        "router_confidence": round(float(rd.confidence), 4),
        "router_prompt_score": round(float(rd.prompt_score), 4),
        "router_prose_score": round(float(rd.prose_score), 4),
        "router_word_count": rd.word_count,
        "router_skipped_detectors": list(rd.skip_detectors),
        "cascade_phases": list(det.cascade_phases),
        "detectors_run": list(det.detectors_run),
        "p_llm": round(float(det.p_llm), 6),
        "label": det.label,
        "conformal_set": det.conformal_set,
        "conformal_alpha": det.conformal_alpha,
        "prediction_set": list(det.prediction_set),
        "confidence_interval": [round(float(x), 4) for x in det.confidence_interval],
        "fusion_mode": det.fusion_mode,
        "calibration_status": det.calibration_status,
        "abstention_reason": det.abstention_reason,
        "override_applied": bool(det.override_applied),
        "coverage_guarantee_applies": bool(det.coverage_guarantee_applies),
        "mixed_report": det.mixed_report,
        "top_features": [[str(n), round(float(v), 4)] for n, v in det.top_features],
        "reason": det.reason,
        "warnings": list(det.warnings or []),
        "detector_errors": list(det.detector_errors or []),
        "phase1_aggregate_p_llm": round(own_cascade.aggregate_p_llm(phase1), 4),
        "aggregate_p_llm_all": round(own_cascade.aggregate_p_llm(results), 4),
        "own_cascade_would_run_phase2": bool(would_p2),
        "own_cascade_would_run_phase3": bool(would_p3),
        "detectors": [detector_row(r) for r in results],
    }
    return row


def run_profile(prof: ProfileInfo, pipeline, samples: list[dict],
                mode: str) -> dict:
    inner = pipeline._inner if isinstance(pipeline, CachedPipeline) else pipeline
    own_cascade = CascadeScheduler(prof.config or {})
    t0 = time.monotonic()
    rows = [
        summarize(prof, s, *pipeline.analyze_detailed(s["text"]), own_cascade)
        for s in samples
    ]
    artifacts = {}
    for label, rel in (("model", prof.model_path), ("conformal", prof.conformal_path),
                       ("detector_calibration", prof.detector_calibration_path)):
        if rel and (ROOT / rel).exists():
            artifacts[label] = {"path": rel, "sha256": _sha256_file(ROOT / rel)}
    dets_cfg = inner._config.get("detectors") or {}
    enabled = [d for d, blk in dets_cfg.items() if (blk or {}).get("enabled", True)]
    return {
        "profile": prof.name,
        "declared_profile": prof.declared_profile,
        "config_path": prof.path,
        "mode": mode,
        "calibration_health": inner.calibration_health(),
        "detector_calibration_state": inner.detector_calibration_state,
        "artifacts": artifacts,
        "enabled_detectors": enabled,
        "seconds": round(time.monotonic() - t0, 2),
        "rows": rows,
    }


def verify_direct(prof: ProfileInfo, phases: tuple[int, ...], samples: list[dict],
                  replay_rows: list[dict]) -> dict:
    """Re-run `prof` end-to-end with CairnPipeline and diff against the replay."""
    pipeline = CairnPipeline(apply_phase_overlay(prof.config or {}, phases))
    per_sample = []
    for sample, rep in zip(samples, replay_rows):
        det, results, _rd = pipeline.analyze_detailed(sample["text"])
        direct_by_id = {r.layer_id: r for r in results}
        replay_by_id = {d["layer_id"]: d for d in rep["detectors"]}
        det_diffs = []
        for det_id, r in direct_by_id.items():
            rp = replay_by_id.get(det_id)
            if rp is None:
                continue
            a, b = _finite(r.p_llm), rp["p_llm"]
            if a is not None and b is not None:
                det_diffs.append(abs(a - b))
        per_sample.append({
            "sample_id": sample["id"],
            "direct_p_llm": round(float(det.p_llm), 6),
            "replay_p_llm": rep["p_llm"],
            "abs_diff": round(abs(float(det.p_llm) - rep["p_llm"]), 8),
            "direct_label": det.label,
            "replay_label": rep["label"],
            "labels_match": det.label == rep["label"],
            "conformal_match": det.conformal_set == rep["conformal_set"],
            "detector_ids_match": set(direct_by_id) == set(replay_by_id),
            "max_detector_p_llm_abs_diff": round(max(det_diffs), 8) if det_diffs else 0.0,
            "direct_detector_errors": list(det.detector_errors or []),
        })
    return {
        "profile": prof.name,
        "n": len(per_sample),
        "max_abs_diff": max((d["abs_diff"] for d in per_sample), default=0.0),
        "max_detector_p_llm_abs_diff": max((d["max_detector_p_llm_abs_diff"] for d in per_sample), default=0.0),
        "all_labels_match": all(d["labels_match"] for d in per_sample),
        "all_conformal_match": all(d["conformal_match"] for d in per_sample),
        "all_detector_ids_match": all(d["detector_ids_match"] for d in per_sample),
        "rows": per_sample,
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "sentence_transformers", "interpret",
                "sklearn", "skops", "numpy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[mod] = None
    try:
        import torch  # noqa: WPS433
        out["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        out["torch_threads"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        out["device"] = None
    return out


def _cell(v):
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) if not isinstance(x, (list, tuple, dict)) else json.dumps(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return v


def write_csvs(out_dir: Path, results: dict[str, dict]) -> tuple[Path, Path]:
    long_path = out_dir / "results_long.csv"
    det_path = out_dir / "detectors_long.csv"
    long_cols = [
        "profile", "sample_id", "domain", "task", "environment", "word_count",
        "router_domain", "router_confidence", "router_prompt_score", "router_prose_score",
        "router_word_count", "router_skipped_detectors", "cascade_phases", "detectors_run",
        "p_llm", "label", "conformal_set", "conformal_alpha", "prediction_set",
        "confidence_interval", "fusion_mode", "calibration_status", "abstention_reason",
        "override_applied", "coverage_guarantee_applies", "top_features",
        "phase1_aggregate_p_llm", "aggregate_p_llm_all",
        "own_cascade_would_run_phase2", "own_cascade_would_run_phase3",
        "detector_errors", "warnings", "reason",
    ]
    with open(long_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=long_cols, extrasaction="ignore")
        w.writeheader()
        for res in results.values():
            for row in res["rows"]:
                w.writerow({k: _cell(row.get(k)) for k in long_cols})
    det_cols = [
        "profile", "sample_id", "layer_id", "phase", "domain", "raw_score", "p_llm",
        "confidence", "determination", "calibrator_applied", "calibrator_flags",
        "skipped_reason", "signals",
    ]
    with open(det_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=det_cols)
        w.writeheader()
        for res in results.values():
            for row in res["rows"]:
                for d in row["detectors"]:
                    w.writerow({
                        "profile": row["profile"], "sample_id": row["sample_id"],
                        **{k: _cell(d.get(k)) for k in det_cols[2:]},
                    })
    return long_path, det_path


def write_xlsx(out_dir: Path, samples: list[dict], results: dict[str, dict],
               profiles: list[ProfileInfo], run_meta: dict) -> Path | None:
    try:
        from openpyxl import Workbook
        from openpyxl.formatting.rule import ColorScaleRule
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("[xlsx] openpyxl not installed; skipping workbook", flush=True)
        return None

    font = Font(name="Arial", size=10)
    bold = Font(name="Arial", size=10, bold=True)
    header_fill = PatternFill("solid", fgColor="DDEBF7")
    label_fill = {
        "RED": "F8CBAD", "AMBER": "FFE699", "YELLOW": "FFF2CC", "GREEN": "C6E0B4",
        "UNCERTAIN": "D9D9D9", "MIXED": "D9D2E9",
    }
    prof_names = list(results.keys())
    sample_ids = [s["id"] for s in samples]

    wb = Workbook()

    def style_sheet(ws, header_rows: int = 1, freeze: str = "B2"):
        for row in ws.iter_rows():
            for c in row:
                c.font = font
        for r in range(1, header_rows + 1):
            for c in ws[r]:
                c.font = bold
                c.fill = header_fill
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = freeze
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(10, width + 2), 60)

    # -- matrices ---------------------------------------------------------
    def matrix_sheet(title: str, key: str, fmt=None, color_scale=False,
                     fill_by_label=False, transform=None):
        ws = wb.create_sheet(title)
        ws.append(["sample_id", "domain", "words", "router"] + prof_names)
        for s in samples:
            first = results[prof_names[0]]["rows"][sample_ids.index(s["id"])] if prof_names else {}
            row = [s["id"], s.get("domain"), s.get("word_count"), first.get("router_domain")]
            for name in prof_names:
                r = results[name]["rows"][sample_ids.index(s["id"])]
                v = r.get(key)
                if transform is not None:
                    v = transform(v)
                row.append(_cell(v) if not isinstance(v, (int, float)) else v)
            ws.append(row)
        style_sheet(ws, freeze="E2")
        n_rows, n_cols = len(samples) + 1, len(prof_names) + 4
        if fmt:
            for row in ws.iter_rows(min_row=2, min_col=5, max_row=n_rows, max_col=n_cols):
                for c in row:
                    c.number_format = fmt
        if color_scale and prof_names:
            rng = f"E2:{get_column_letter(n_cols)}{n_rows}"
            ws.conditional_formatting.add(rng, ColorScaleRule(
                start_type="num", start_value=0, start_color="C6E0B4",
                mid_type="num", mid_value=0.5, mid_color="FFF2CC",
                end_type="num", end_value=1, end_color="F8CBAD",
            ))
        if fill_by_label:
            for row in ws.iter_rows(min_row=2, min_col=5, max_row=n_rows, max_col=n_cols):
                for c in row:
                    color = label_fill.get(str(c.value))
                    if color:
                        c.fill = PatternFill("solid", fgColor=color)
        return ws

    # -- summary (static values: this is an export of pipeline outputs, and
    #    the matrices it summarises never change inside the workbook) -------
    ws = wb.active
    ws.title = "Summary"
    ws.append(["profile", "mode", "calibration_status", "specdetect",
               "mean p(LLM)", "median p(LLM)", "n RED", "n AMBER", "n YELLOW", "n GREEN",
               "n UNCERTAIN", "n MIXED", "conformal [LLM]", "conformal [human]",
               "conformal abstain", "own cascade would escalate to phase 2"])
    n = len(samples)
    for name in prof_names:
        res = results[name]
        rows = res["rows"]
        ps = sorted(r["p_llm"] for r in rows)
        labels = [r["label"] for r in rows]
        conf = [tuple(r["conformal_set"] or []) for r in rows]
        median = (ps[len(ps) // 2] if len(ps) % 2 else (ps[len(ps) // 2 - 1] + ps[len(ps) // 2]) / 2) if ps else None
        ws.append([
            name, res["mode"], res["calibration_health"]["calibration_status"],
            "yes" if "specdetect" in res["enabled_detectors"] else "no",
            sum(ps) / len(ps) if ps else None, median,
            labels.count("RED"), labels.count("AMBER"), labels.count("YELLOW"),
            labels.count("GREEN"), labels.count("UNCERTAIN"), labels.count("MIXED"),
            conf.count(("LLM",)), conf.count(("human",)),
            sum(1 for c in conf if len(c) == 2),
            sum(1 for r in rows if r["own_cascade_would_run_phase2"]),
        ])
    ws.append([])
    ws.append([f"n samples = {n}; phases run = {run_meta['phases']}. Static export: every cell in "
               "this workbook is a CAIRN pipeline output written by scripts/run_profile_screen.py "
               "(no formulas); the input carries no ground-truth labels."])
    ws.append(["See README.md next to this workbook for the method and caveats."])
    style_sheet(ws, freeze="B2")
    for row in ws.iter_rows(min_row=2, min_col=5, max_row=len(prof_names) + 1, max_col=6):
        for c in row:
            c.number_format = "0.000"

    matrix_sheet("p_llm", "p_llm", fmt="0.000", color_scale=True)
    matrix_sheet("label", "label", fill_by_label=True)
    matrix_sheet("conformal_set", "conformal_set")
    matrix_sheet("own_cascade_phase2", "own_cascade_would_run_phase2",
                 transform=lambda v: "yes" if v else "no")

    # -- long tables ------------------------------------------------------
    ws = wb.create_sheet("results_long")
    cols = ["profile", "sample_id", "domain", "task", "environment", "word_count",
            "router_domain", "cascade_phases", "p_llm", "label", "conformal_set",
            "prediction_set", "confidence_interval", "fusion_mode", "calibration_status",
            "abstention_reason", "override_applied", "top_features",
            "phase1_aggregate_p_llm", "own_cascade_would_run_phase2",
            "own_cascade_would_run_phase3", "detector_errors", "reason"]
    ws.append(cols)
    for res in results.values():
        for row in res["rows"]:
            ws.append([_cell(row.get(k)) for k in cols])
    style_sheet(ws, freeze="C2")

    ws = wb.create_sheet("detectors_long")
    cols = ["profile", "sample_id", "layer_id", "phase", "raw_score", "p_llm", "confidence",
            "determination", "calibrator_applied", "calibrator_flags", "skipped_reason"]
    ws.append(cols)
    for res in results.values():
        for row in res["rows"]:
            for d in row["detectors"]:
                ws.append([row["profile"], row["sample_id"]] + [_cell(d.get(k)) for k in cols[2:]])
    style_sheet(ws, freeze="D2")

    ws = wb.create_sheet("profiles")
    cols = ["name", "status", "declared_profile", "path", "extends", "model_path",
            "conformal_path", "detector_calibration_path", "missing", "note"]
    ws.append(cols)
    for p in profiles:
        d = p.public()
        ws.append([_cell(d.get(k)) for k in cols])
    style_sheet(ws, freeze="B2")

    ws = wb.create_sheet("run")
    ws.append(["key", "value"])
    for k, v in run_meta.items():
        ws.append([k, _cell(v) if not isinstance(v, (int, float)) else v])
    style_sheet(ws, freeze="A2")

    path = out_dir / "results.xlsx"
    wb.save(path)
    return path


def write_markdown(out_dir: Path, samples: list[dict], results: dict[str, dict],
                   profiles: list[ProfileInfo], run_meta: dict, raw_info: dict,
                   verification: dict | None) -> Path:
    """Auto-generated tables (matrices + availability) as results.md."""
    names = list(results.keys())
    lines: list[str] = []
    add = lines.append
    add("# Profile screen — generated tables")
    add("")
    add(f"Generated {run_meta['timestamp_utc']} from `{Path(run_meta['dataset']).name}` "
        f"(n={run_meta['n_samples']}, phases {run_meta['phases']}, commit "
        f"`{(run_meta.get('git_commit') or 'n/a')[:12]}`). Every number is a pipeline "
        "output; the input carries no ground-truth labels.")
    add("")
    add("## Profiles run")
    add("")
    add("| profile | mode | calibration_status | detector calibration | specdetect | mean p(LLM) | "
        "RED | AMBER | YELLOW | GREEN | UNCERTAIN | MIXED | conf [LLM] | conf [human] | conf abstain | "
        "own cascade -> phase 2 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        res = results[name]
        rows = res["rows"]
        labels = [r["label"] for r in rows]
        conf = [tuple(r["conformal_set"] or []) for r in rows]
        mean_p = sum(r["p_llm"] for r in rows) / max(len(rows), 1)
        add(
            f"| `{name}` | {res['mode']} | {res['calibration_health']['calibration_status']} | "
            f"{res['detector_calibration_state']['state']} | "
            f"{'yes' if 'specdetect' in res['enabled_detectors'] else 'no'} | {mean_p:.3f} | "
            + " | ".join(str(labels.count(l)) for l in ("RED", "AMBER", "YELLOW", "GREEN", "UNCERTAIN", "MIXED"))
            + f" | {conf.count(('LLM',))} | {conf.count(('human',))} | "
            f"{sum(1 for c in conf if len(c) == 2)} | "
            f"{sum(1 for r in rows if r['own_cascade_would_run_phase2'])}/{len(rows)} |"
        )
    add("")

    def matrix(title: str, key: str, fmt):
        add(f"## {title}")
        add("")
        add("| sample | domain | words | router | " + " | ".join(f"`{n}`" for n in names) + " |")
        add("|---|---|---|---|" + "---|" * len(names))
        for i, s in enumerate(samples):
            first = results[names[0]]["rows"][i] if names else {}
            cells = [fmt(results[n]["rows"][i].get(key)) for n in names]
            add(f"| {s['id']} | {s.get('domain', '')} | {s.get('word_count', '')} | "
                f"{first.get('router_domain', '')} | " + " | ".join(cells) + " |")
        add("")

    matrix("p(LLM) per sample x profile", "p_llm", lambda v: f"{v:.3f}")
    matrix("Label per sample x profile", "label", lambda v: str(v))
    matrix("Conformal set per sample x profile", "conformal_set",
           lambda v: "+".join(v) if v else "-")
    matrix("Own-cascade escalation to phase 2 (counterfactual)", "own_cascade_would_run_phase2",
           lambda v: "yes" if v else "no")

    add("## Raw detector pass")
    add("")
    unavailable = [r for r in raw_info.get("detector_availability", [])
                   if r["enabled"] and not r["available"]]
    add(f"Superset detectors: {', '.join(run_meta['superset_enabled_detectors'])}.")
    add("")
    if unavailable:
        add("Enabled but unavailable: " + "; ".join(f"{r['id']} ({r['reason']})" for r in unavailable))
    else:
        add("All enabled detectors were available.")
    if raw_info.get("detector_errors"):
        add("")
        add("Detector errors during the raw pass:")
        for sid, errs in raw_info["detector_errors"].items():
            add(f"- {sid}: {errs}")
    add("")
    add("## Replay verification")
    add("")
    if verification:
        add(f"`{verification['profile']}` re-run directly with `CairnPipeline` on "
            f"{verification['n']} sample(s): max |p(LLM) diff| = {verification['max_abs_diff']}, "
            f"max per-detector |p diff| = {verification['max_detector_p_llm_abs_diff']}, "
            f"labels match = {verification['all_labels_match']}, conformal sets match = "
            f"{verification['all_conformal_match']}, detector sets match = "
            f"{verification['all_detector_ids_match']}.")
    else:
        add("Not run (pass --verify-direct <profile>).")
    add("")
    add("## Profile availability")
    add("")
    add("| profile | status | config | detail |")
    add("|---|---|---|---|")
    for p in profiles:
        detail = "; ".join(p.missing) if p.missing else p.note
        add(f"| `{p.name}` | {p.status} | `{p.path}` | {detail} |")
    add("")
    path = out_dir / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def print_matrix(samples: list[dict], results: dict[str, dict]) -> None:
    names = list(results.keys())
    if not names:
        return
    short = [n.replace("master_minus_arxiv", "mma") for n in names]
    width = max(len(s) for s in short) if short else 10
    width = max(width, 7)
    print("\n" + "=" * (12 + (width + 1) * len(names)))
    print("p(LLM) per sample x profile" + "   (label in the second block)")
    print("=" * (12 + (width + 1) * len(names)))
    print(f"{'sample':10s} " + " ".join(f"{s:>{width}s}" for s in short))
    for i, s in enumerate(samples):
        vals = [results[n]["rows"][i]["p_llm"] for n in names]
        print(f"{s['id']:10s} " + " ".join(f"{v:>{width}.3f}" for v in vals))
    print("-" * (12 + (width + 1) * len(names)))
    for i, s in enumerate(samples):
        vals = [results[n]["rows"][i]["label"] for n in names]
        print(f"{s['id']:10s} " + " ".join(f"{v:>{width}s}" for v in vals))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--phases", default="1,2", help="cascade phases to run: 1 | 1,2 | 1,2,3")
    ap.add_argument("--profiles", default=None,
                    help="comma-separated profile names to run (default: every available)")
    ap.add_argument("--features-cache", type=Path, default=None,
                    help="raw feature cache path (default: data/<dataset-stem>_features_p<phases>.jsonl)")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N samples")
    ap.add_argument("--verify-direct", default=None,
                    help="profile name to re-run end-to-end with CairnPipeline and diff against replay")
    ap.add_argument("--verify-limit", type=int, default=None)
    ap.add_argument("--xlsx", action="store_true", help="also write results.xlsx")
    ap.add_argument("--dry-run", action="store_true",
                    help="print profile availability, the superset detector set and replay "
                         "compatibility, then exit without running any detector")
    args = ap.parse_args(argv)

    # Resolve caller-relative paths first: profile artifact paths inside the
    # configs are repo-relative, so the run itself happens from ROOT.
    args.dataset = args.dataset.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.features_cache is not None:
        args.features_cache = args.features_cache.resolve()
    os.chdir(ROOT)
    phases = parse_phases(args.phases)
    samples = load_samples(args.dataset, args.limit)
    if not samples:
        raise SystemExit("dataset is empty")
    ids = [s["id"] for s in samples]
    if len(set(ids)) != len(ids):
        raise SystemExit("dataset ids must be unique")

    profiles = discover_profiles()
    available = [p for p in profiles if p.status == "available"]
    if args.profiles:
        wanted = {x.strip() for x in args.profiles.split(",") if x.strip()}
        unknown = wanted - {p.name for p in profiles}
        if unknown:
            raise SystemExit(f"unknown profile(s): {sorted(unknown)}")
        not_avail = [p for p in profiles if p.name in wanted and p.status != "available"]
        for p in not_avail:
            print(f"[profiles] {p.name} requested but {p.status}: {p.missing or p.note}", flush=True)
        available = [p for p in available if p.name in wanted]
    print(f"[profiles] {len(profiles)} configs discovered; "
          f"{sum(1 for p in profiles if p.status == 'available')} available; "
          f"running {len(available)}: {[p.name for p in available]}", flush=True)
    if not available:
        raise SystemExit("no available calibrated profiles to run")

    superset_cfg = build_superset_config(available, phases)
    if args.dry_run:
        for p in profiles:
            detail = "; ".join(p.missing) if p.missing else p.note
            print(f"  {p.status:15s} {p.name:45s} {detail}")
        enabled = sorted(d for d, blk in (superset_cfg.get("detectors") or {}).items()
                         if (blk or {}).get("enabled", True))
        print(f"[superset] phases={list(phases)} enabled detectors={enabled}")
        print(f"[superset] cascade={superset_cfg.get('cascade')}")
        for prof in available:
            diffs = replay_incompatibilities(apply_phase_overlay(prof.config or {}, phases),
                                             superset_cfg, phases)
            print(f"  {prof.name:45s} {'replay' if not diffs else 'direct: ' + str(diffs)}")
        return 0
    cache_path = args.features_cache or (
        ROOT / "data" / f"{args.dataset.stem}_features_p{''.join(map(str, phases))}.jsonl"
    )
    t_run0 = time.monotonic()
    raw_info = build_raw_cache(samples, superset_cfg, cache_path, rebuild=args.rebuild_cache)

    results: dict[str, dict] = {}
    load_failures: dict[str, str] = {}
    for prof in available:
        cfg = apply_phase_overlay(prof.config or {}, phases)
        diffs = replay_incompatibilities(cfg, superset_cfg, phases)
        try:
            if diffs:
                print(f"[{prof.name}] config differs from superset on {diffs}; running directly", flush=True)
                pipeline = CairnPipeline(cfg)
                mode = "direct"
            else:
                pipeline = CachedPipeline(cache_path, config=cfg)
                mode = "replay"
        except Exception as exc:  # noqa: BLE001 - keep going, report per profile
            load_failures[prof.name] = f"{type(exc).__name__}: {exc}"
            print(f"[{prof.name}] pipeline construction FAILED: {exc}", flush=True)
            continue
        res = run_profile(prof, pipeline, samples, mode)
        res["replay_incompatibilities"] = diffs
        results[prof.name] = res
        health = res["calibration_health"]
        labels = [r["label"] for r in res["rows"]]
        print(
            f"[{prof.name}] {mode} status={health['calibration_status']} "
            f"detcal={res['detector_calibration_state']['state']} "
            f"labels={ {l: labels.count(l) for l in sorted(set(labels))} } "
            f"mean_p={sum(r['p_llm'] for r in res['rows']) / len(res['rows']):.3f} "
            f"({res['seconds']}s)",
            flush=True,
        )

    verification = None
    if args.verify_direct:
        if args.verify_direct not in results:
            raise SystemExit(f"--verify-direct: {args.verify_direct} was not run")
        if results[args.verify_direct]["mode"] != "replay":
            print(f"[verify] {args.verify_direct} already ran directly; nothing to compare", flush=True)
        else:
            prof = next(p for p in available if p.name == args.verify_direct)
            n = args.verify_limit or len(samples)
            print(f"[verify] re-running {prof.name} directly on {n} sample(s)...", flush=True)
            verification = verify_direct(prof, phases, samples[:n],
                                         results[prof.name]["rows"][:n])
            print(
                f"[verify] max |p_llm diff|={verification['max_abs_diff']} "
                f"max detector diff={verification['max_detector_p_llm_abs_diff']} "
                f"labels match={verification['all_labels_match']} "
                f"conformal match={verification['all_conformal_match']}",
                flush=True,
            )

    run_meta = {
        "timestamp_utc": _now_iso(),
        "git_commit": _git_commit(),
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256_file(args.dataset),
        "n_samples": len(samples),
        "phases": list(phases),
        "features_cache": str(cache_path),
        "superset_enabled_detectors": sorted(
            d for d, blk in (superset_cfg.get("detectors") or {}).items()
            if (blk or {}).get("enabled", True)
        ),
        "superset_cascade": superset_cfg.get("cascade"),
        "profiles_run": list(results.keys()),
        "profiles_load_failed": load_failures,
        "versions": _versions(),
        "total_seconds": round(time.monotonic() - t_run0, 1),
    }
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": run_meta,
        "samples": [
            {**_sample_meta(s), "sha256": text_hash(normalize_text(s["text"])),
             "excerpt": normalize_text(s["text"])[:120]}
            for s in samples
        ],
        "profiles": [p.public() for p in profiles],
        "raw_pass": raw_info,
        "results": results,
        "verification": verification,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                          encoding="utf-8")
    (out_dir / "profiles.json").write_text(
        json.dumps([p.public() for p in profiles], indent=2) + "\n", encoding="utf-8")
    long_path, det_path = write_csvs(out_dir, results)
    md_path = write_markdown(out_dir, samples, results, profiles, run_meta, raw_info, verification)
    xlsx_path = write_xlsx(out_dir, samples, results, profiles, run_meta) if args.xlsx else None

    print_matrix(samples, results)
    print(f"\nwrote {out_dir / 'results.json'}, {long_path}, {det_path}, {md_path}, "
          f"{out_dir / 'profiles.json'}" + (f", {xlsx_path}" if xlsx_path else ""))
    if load_failures:
        print(f"profiles that failed to load: {load_failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
