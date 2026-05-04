import json
import os
import subprocess
import sys
from pathlib import Path
import pytest


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


def test_holdout_source_excludes_family_from_train(tmp_path):
    pytest.importorskip("interpret")
    ds = tmp_path / "ds.jsonl"
    _write_dataset(ds)
    out = tmp_path / "ebm.pkl"
    # Resolve config relative to the repo root so the test does not silently
    # break if pytest is invoked from a different cwd.
    repo_root = Path(__file__).resolve().parent.parent
    cfg = repo_root / "configs" / "default.yaml"
    # Ensure the subprocess can import `cairn` even when the editable install's
    # .pth file points elsewhere — fall back to the repo root on PYTHONPATH.
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
    )
    res = subprocess.run(
        [sys.executable, "scripts/train_fusion.py",
         "--dataset", str(ds), "--config", str(cfg),
         "--out", str(out), "--holdout-source", "openai"],
        capture_output=True, text=True, env=env, cwd=str(repo_root),
    )
    assert res.returncode == 0, res.stderr
    assert "trained on 30" in res.stdout
    assert "held out 15" in res.stdout
