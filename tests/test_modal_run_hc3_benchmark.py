"""Pure-helper tests for scripts/modal_run_hc3_benchmark.py.

The remote `run_grid` function and the `local_entrypoint` driver need
Modal credentials to actually execute. The pure helpers
(`parse_csv_list`, `pack_directory`, `unpack_tarball`, `collect_outputs`)
do not — they only use `io`, `tarfile`, and `pathlib`. We stub
`sys.modules['modal']` with a MagicMock so the script module loads in
environments where `pip install modal` has not been run; the stub never
touches the helpers under test.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "modal_run_hc3_benchmark.py"


def _load_script_module() -> types.ModuleType:
    """Load modal_run_hc3_benchmark.py with a stubbed `modal` package.

    The script's top-level decorators (`@app.function(...)`,
    `@app.local_entrypoint()`) execute at import time. MagicMock returns
    child mocks for arbitrary attribute access and call — that is enough
    to satisfy the chained `modal.Image.debian_slim(...).pip_install(...)
    .add_local_dir(...)` builder, the `modal.Volume.from_name` calls, and
    the decorators themselves.
    """
    name = "scripts_modal_run_hc3_benchmark"
    if name in sys.modules:
        return sys.modules[name]
    sys.modules.setdefault("modal", MagicMock(name="modal"))
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mh():
    return _load_script_module()


def test_parse_csv_list_handles_empty_and_whitespace(mh):
    assert mh.parse_csv_list("") == []
    assert mh.parse_csv_list(None) == []
    assert mh.parse_csv_list("finance") == ["finance"]
    assert mh.parse_csv_list("finance, medicine ,open_qa") == [
        "finance", "medicine", "open_qa",
    ]
    # Empty fragments between commas are dropped.
    assert mh.parse_csv_list("a,,b,") == ["a", "b"]


def test_pack_unpack_round_trip(mh, tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.txt").write_text("hello")
    nested = root / "sub" / "deep"
    nested.mkdir(parents=True)
    (nested / "b.json").write_text('{"k": 1}')

    files = [root / "a.txt", nested / "b.json"]
    blob = mh.pack_directory(root, files)
    assert isinstance(blob, bytes) and len(blob) > 0

    dest = tmp_path / "dest"
    written = mh.unpack_tarball(blob, dest)
    assert {p.relative_to(dest).as_posix() for p in written} == {
        "a.txt", "sub/deep/b.json",
    }
    assert (dest / "a.txt").read_text() == "hello"
    assert (dest / "sub" / "deep" / "b.json").read_text() == '{"k": 1}'


def test_unpack_rejects_path_traversal(mh, tmp_path):
    """Tar entries with `..` segments must be refused, not silently extracted."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="../escape.txt")
        payload = b"x"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe"):
        mh.unpack_tarball(buf.getvalue(), tmp_path / "dest")


def test_unpack_rejects_absolute_paths(mh, tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="/etc/passwd")
        payload = b"x"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe"):
        mh.unpack_tarball(buf.getvalue(), tmp_path / "dest")


def test_collect_outputs_walks_dir_and_includes_cache(mh, tmp_path):
    out_dir = tmp_path / "out"
    (out_dir / "a").mkdir(parents=True)
    (out_dir / "a" / "x.json").write_text("{}")
    (out_dir / "summary.csv").write_text("subset,config\n")
    cache = tmp_path / "features.jsonl"
    cache.write_text("{}\n")

    files = mh.collect_outputs(out_dir, cache)
    rel = {p.relative_to(tmp_path).as_posix() for p in files}
    assert rel == {"out/a/x.json", "out/summary.csv", "features.jsonl"}


def test_collect_outputs_handles_missing_dir_and_cache(mh, tmp_path):
    files = mh.collect_outputs(tmp_path / "does-not-exist", None)
    assert files == []
    files = mh.collect_outputs(tmp_path / "does-not-exist", tmp_path / "also-missing.jsonl")
    assert files == []


def test_module_exposes_modal_app_constants(mh):
    """Sanity: top-level objects are present (decorators executed against the stub)."""
    assert hasattr(mh, "app")
    assert hasattr(mh, "image")
    assert hasattr(mh, "run_grid")
    assert hasattr(mh, "main")
