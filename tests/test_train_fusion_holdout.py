import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cairn.contracts import LayerResult, RouterDecision
from cairn.feature_cache import text_hash, write_record
from cairn.pipeline import normalize_text


def _write_dataset(path: Path):
    rows = []
    for i in range(15):
        rows.append({"id": f"a{i}", "text": "Anthropic LLM sample. " * 30,
                     "label": 1, "source": "anthropic"})
    for i in range(15):
        rows.append({"id": f"o{i}", "text": "OpenAI LLM sample. " * 30,
                     "label": 1, "source": "openai"})
    for i in range(15):
        rows.append({"id": f"h{i}", "text": "Human writing sample. " * 30,
                     "label": 0, "source": "hc3"})
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
    )
    return env


def _write_cached_train_all_fixture(dataset: Path, cache: Path) -> None:
    rows: list[dict[str, object]] = []
    with cache.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(30):
            label = index % 2
            text = " ".join([f"sample-{index}"] * 600)
            rows.append(
                {
                    "id": f"row-{index}",
                    "text": text,
                    "label": label,
                    "source": "synthetic-llm" if label else "synthetic-human",
                }
            )
            p_llm = 0.9 if label else 0.1
            result = LayerResult(
                layer_id="preamble",
                domain="universal",
                raw_score=p_llm,
                p_llm=p_llm,
                confidence=1.0,
                signals={
                    "severity": "HIGH" if label else "NONE",
                    "n_matches": 3 if label else 0,
                },
                determination="RED" if label else "GREEN",
                attacker_tiers=["A0"],
                compute_cost="trivial",
                min_text_length=0,
            )
            router = RouterDecision(
                domain="prompt",
                confidence=0.95,
                prompt_score=0.9,
                prose_score=0.1,
                word_count=600,
                recommended_detectors=["preamble"],
                skip_detectors=[],
            )
            normalized = normalize_text(text)
            write_record(handle, text_hash(normalized), [result], router)
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_holdout_source_excludes_family_from_train(tmp_path):
    pytest.importorskip("interpret")
    ds = tmp_path / "ds.jsonl"
    _write_dataset(ds)
    out = tmp_path / "ebm.pkl"
    # Resolve config relative to the repo root so the test does not silently
    # break if pytest is invoked from a different cwd.
    repo_root = _repo_root()
    cfg = repo_root / "configs" / "default.yaml"
    # Ensure the subprocess can import `cairn` even when the editable install's
    # .pth file points elsewhere — fall back to the repo root on PYTHONPATH.
    env = _subprocess_env(repo_root)
    res = subprocess.run(
        [sys.executable, "scripts/train_fusion.py",
         "--dataset", str(ds), "--config", str(cfg),
         "--out", str(out), "--holdout-source", "openai"],
        capture_output=True, text=True, env=env, cwd=str(repo_root),
    )
    assert res.returncode == 0, res.stderr
    assert "trained on 30" in res.stdout
    assert "held out 15" in res.stdout


def test_training_assembler_uses_profile_word_count_clamp():
    from scripts.train_fusion import build_training_assembler

    config = {"fusion": {"word_count_clamp": 200}}
    assembler = build_training_assembler(config, drop_features=set())
    vector = assembler.assemble([], word_count=600, domain="prompt")

    assert vector["word_count"] == 200.0
    assert vector["log_word_count"] == pytest.approx(math.log1p(200))
    assert vector["is_medium"] == 1.0
    assert vector["is_long"] == 0.0


def test_train_all_uses_every_row_and_writes_complete_metadata(tmp_path):
    pytest.importorskip("interpret")
    pytest.importorskip("skops")
    dataset = tmp_path / "train-all.jsonl"
    cache = tmp_path / "train-all.features.jsonl"
    _write_cached_train_all_fixture(dataset, cache)
    config = tmp_path / "clamped.yaml"
    config.write_text(
        "_extends: default\n"
        "fusion:\n"
        "  word_count_clamp: 200\n",
        encoding="utf-8",
    )
    out = tmp_path / "ebm.skops"
    repo_root = _repo_root()

    res = subprocess.run(
        [
            sys.executable,
            "scripts/train_fusion.py",
            "--dataset",
            str(dataset),
            "--config",
            str(config),
            "--out",
            str(out),
            "--features-cache",
            str(cache),
            "--train-all",
            "--interactions",
            "0",
            "--drop-features",
            "specdetect_total_energy",
        ],
        capture_output=True,
        text=True,
        env=_subprocess_env(repo_root),
        cwd=str(repo_root),
    )

    assert res.returncode == 0, res.stderr
    assert "trained on 30, held out 0" in res.stdout
    metadata = json.loads(
        out.with_suffix(out.suffix + ".meta.json").read_text(encoding="utf-8")
    )
    required = {
        "dataset_path",
        "dataset_sha256",
        "features_cache_path",
        "features_cache_sha256",
        "n_train",
        "n_test",
        "label_balance_train",
        "label_balance_test",
        "trained_at",
        "interactions",
        "max_rounds",
        "random_state",
        "python_version",
        "numpy_version",
        "sklearn_version",
        "interpret_version",
        "skops_version",
        "feature_names",
        "dropped_features",
        "schema_version",
        "word_count_clamp",
    }
    assert required <= metadata.keys()
    assert metadata["dataset_path"] == str(dataset)
    assert metadata["dataset_sha256"] == hashlib.sha256(
        dataset.read_bytes()
    ).hexdigest()
    assert len(metadata["dataset_sha256"]) == 64
    assert metadata["features_cache_path"] == str(cache)
    assert metadata["features_cache_sha256"] == hashlib.sha256(
        cache.read_bytes()
    ).hexdigest()
    assert len(metadata["features_cache_sha256"]) == 64
    assert metadata["n_train"] == 30
    assert metadata["n_test"] == 0
    assert metadata["label_balance_train"] == pytest.approx(0.5)
    assert metadata["label_balance_test"] is None
    assert metadata["interactions"] == 0
    assert metadata["random_state"] == 42
    assert metadata["word_count_clamp"] == 200
    assert metadata["dropped_features"] == ["specdetect_total_energy"]
    assert "specdetect_total_energy" not in metadata["feature_names"]
    assert metadata["feature_names"]
    for version_key in (
        "python_version",
        "numpy_version",
        "sklearn_version",
        "interpret_version",
        "skops_version",
    ):
        assert isinstance(metadata[version_key], str) and metadata[version_key]


def test_train_all_is_mutually_exclusive_with_holdout_source(tmp_path):
    repo_root = _repo_root()
    res = subprocess.run(
        [
            sys.executable,
            "scripts/train_fusion.py",
            "--dataset",
            str(tmp_path / "missing.jsonl"),
            "--config",
            str(tmp_path / "missing.yaml"),
            "--out",
            str(tmp_path / "unused.skops"),
            "--train-all",
            "--holdout-source",
            "openai",
        ],
        capture_output=True,
        text=True,
        env=_subprocess_env(repo_root),
        cwd=str(repo_root),
    )

    assert res.returncode == 2
    assert "not allowed with argument" in res.stderr
