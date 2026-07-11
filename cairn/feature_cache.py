"""On-disk cache of pipeline.analyze_detailed outputs, keyed by text hash.

Building the cache once and replaying it is dramatically faster than
re-running the full detector pipeline on every calibration script
invocation (calibrate_detectors -> train_fusion -> calibrate_conformal
-> run_benchmark, plus N×M ablation re-runs). The cache stores the
calibrator-INDEPENDENT detector outputs (raw_score, signals, etc.); each
caller's own calibrator is applied at read-time so different ablations
can swap calibrators against the same cached features.

Schema (JSONL, one record per sample):
    {"hash": "<sha256(text)>", "results": [<LayerResult dict>...],
     "router": {<RouterDecision dict>}}

Records are keyed by sha256(text) so the lookup matches the
CairnPipeline.analyze_detailed(text) call signature exactly — no API
change required in calling scripts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from cairn.contracts import Determination, LayerResult, RouterDecision
from cairn.pipeline import CairnPipeline, normalize_text


def text_hash(text: str) -> str:
    """sha256 hex of the (already normalized) text. Used as the cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _result_to_dict(r: LayerResult) -> dict:
    return asdict(r)


def _result_from_dict(d: dict) -> LayerResult:
    return LayerResult(**d)


def _router_to_dict(rd: RouterDecision) -> dict:
    return asdict(rd)


def _router_from_dict(d: dict) -> RouterDecision:
    return RouterDecision(**d)


def write_record(out_handle, key: str, results: list[LayerResult],
                 rd: RouterDecision) -> None:
    record = {
        "hash": key,
        "results": [_result_to_dict(r) for r in results],
        "router": _router_to_dict(rd),
    }
    out_handle.write(json.dumps(record) + "\n")


def load_cache(path: Path) -> dict[str, dict]:
    """Returns {text_hash: {results: [...], router: {...}}}."""
    cache: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cache[r["hash"]] = {"results": r["results"], "router": r["router"]}
    return cache


class CachedPipeline:
    """Drop-in replacement for CairnPipeline that reads detector outputs
    from a pre-built cache and applies the inner pipeline's calibrator,
    fusion, and decision logic at read-time.

    Same `analyze_detailed(text)` signature as CairnPipeline so calling
    scripts swap pipeline instances and otherwise stay unchanged.
    """

    def __init__(self, cache_path: Path, config_path: Path) -> None:
        self._inner = CairnPipeline.from_config_file(config_path)
        self._cache = load_cache(cache_path)

    def analyze_detailed(
        self, text: str, task_metadata: dict | None = None
    ) -> tuple[Determination, list[LayerResult], RouterDecision]:
        text = normalize_text(text)
        key = text_hash(text)
        cached = self._cache.get(key)
        if cached is None:
            raise KeyError(
                f"text not present in feature cache (hash={key[:12]}...). "
                f"Rebuild the cache to include this dataset."
            )
        rd = _router_from_dict(cached["router"])
        raw_results = cached["results"]
        detector_config = self._inner._config.get("detectors", {})
        router_skips = set(rd.skip_detectors)
        cal = self._inner._detector_calibrator

        def replay_phase(phase: int) -> list[LayerResult]:
            phase_results: list[LayerResult] = []
            for detector_id in self._inner._cascade.detectors_for_phase(phase):
                if detector_id in router_skips:
                    continue
                config = detector_config.get(detector_id, {}) or {}
                if not config.get("enabled", True):
                    continue
                for payload in raw_results:
                    if payload.get("layer_id") != detector_id:
                        continue
                    result = _result_from_dict(payload)
                    if cal is not None:
                        cal.apply(result)
                    phase_results.append(result)
            return phase_results

        phases_run = [1]
        results = replay_phase(1)
        if self._inner._cascade.should_run_phase2(results):
            results.extend(replay_phase(2))
            phases_run.append(2)
            if self._inner._cascade.should_run_phase3(results):
                results.extend(replay_phase(3))
                phases_run.append(3)

        fusion_result = self._inner._fusion.fuse(
            results, word_count=rd.word_count, domain=rd.domain,
        )
        determination = self._inner._decision.decide(fusion_result, results)
        determination.calibration_status = self._inner._calibration_status
        determination.cascade_phases = phases_run
        determination.detector_errors = []
        return determination, results, rd

    def analyze(self, text: str, task_metadata: dict | None = None) -> Determination:
        det, _r, _rd = self.analyze_detailed(text, task_metadata)
        return det
