from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from cairn.contracts import LayerResult, RouterDecision
from cairn.feature_cache import text_hash, write_record
from cairn.pipeline import normalize_text


def _write_cached_fixture(dataset: Path, cache: Path) -> None:
    rows: list[dict[str, object]] = []
    with cache.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(40):
            label = index % 2
            text = " ".join([f"conformal-{index}"] * 100)
            rows.append(
                {
                    "id": f"cal-{index}",
                    "text": text,
                    "label": label,
                    "source": "synthetic-llm" if label else "synthetic-human",
                }
            )
            offset = (index // 2) / 100.0
            p_llm = (0.8 + offset) if label else (0.1 + offset)
            result = LayerResult(
                layer_id="preamble",
                domain="universal",
                raw_score=p_llm,
                p_llm=p_llm,
                confidence=1.0,
                signals={"severity": "HIGH" if label else "NONE"},
                determination="RED" if label else "GREEN",
                attacker_tiers=["A0"],
                compute_cost="trivial",
                min_text_length=0,
            )
            router = RouterDecision(
                domain="prose",
                confidence=0.95,
                prompt_score=0.1,
                prose_score=0.9,
                word_count=100,
                recommended_detectors=["preamble"],
                skip_detectors=[],
            )
            write_record(
                handle,
                text_hash(normalize_text(text)),
                [result],
                router,
            )
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_manifest_records_hashed_cached_calibration_provenance(tmp_path):
    dataset = tmp_path / "calibration.jsonl"
    cache = tmp_path / "calibration.features.jsonl"
    _write_cached_fixture(dataset, cache)
    out = tmp_path / "conformal.json"
    manifest_path = tmp_path / "conformal.manifest.json"
    repo_root = Path(__file__).resolve().parent.parent
    config = repo_root / "configs" / "default.yaml"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_conformal.py",
            "--dataset",
            str(dataset),
            "--config",
            str(config),
            "--features-cache",
            str(cache),
            "--alpha",
            "0.1",
            "--out",
            str(out),
            "--manifest-out",
            str(manifest_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )

    assert result.returncode == 0, result.stderr
    assert out.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    assert len(dataset_hash) == len(cache_hash) == 64
    assert manifest["dataset_path"] == str(dataset)
    assert manifest["dataset_sha256"] == dataset_hash
    assert manifest["features_cache_path"] == str(cache)
    assert manifest["features_cache_sha256"] == cache_hash
    assert manifest["alpha"] == 0.1
    assert manifest["n_calibration"] == 40
    assert manifest["class_counts"] == {"0": 20, "1": 20}
    assert manifest["router_counts"] == {"prose": 40}
    assert set(manifest["threshold_per_class"]) == {"0", "1"}
    assert set(manifest["threshold_per_router"]["prose"]) == {"0", "1"}
    assert manifest["python_version"]
    assert manifest["numpy_version"]
    assert manifest["sklearn_version"]
    assert manifest["calibrated_at"].endswith("+00:00")
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


def test_manifest_out_cannot_alias_conformal_output(tmp_path):
    dataset = tmp_path / "calibration.jsonl"
    cache = tmp_path / "calibration.features.jsonl"
    _write_cached_fixture(dataset, cache)
    out = tmp_path / "shared-output.json"
    sentinel = "existing output must survive\n"
    out.write_text(sentinel, encoding="utf-8")
    repo_root = Path(__file__).resolve().parent.parent
    config = repo_root / "configs" / "default.yaml"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_conformal.py",
            "--dataset",
            str(dataset),
            "--config",
            str(config),
            "--features-cache",
            str(cache),
            "--out",
            str(out),
            "--manifest-out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )

    assert result.returncode == 2
    assert "--manifest-out must differ from --out" in result.stderr
    assert out.read_text(encoding="utf-8") == sentinel
