"""Audit chain tests."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cairn.console import main as cairn_cli
from cairn.pipeline import CairnPipeline
from cairn.provenance.chain import AuditChain, GENESIS


def test_first_entry_genesis(tmp_path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    first = chain.append({"submission_id": "a1"})
    assert first["_prev_hash"] == GENESIS
    assert "_hash" in first


def test_chain_links_forward(tmp_path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    e1 = chain.append({"submission_id": "a1"})
    e2 = chain.append({"submission_id": "a2"})
    e3 = chain.append({"submission_id": "a3"})
    assert e2["_prev_hash"] == e1["_hash"]
    assert e3["_prev_hash"] == e2["_hash"]


def test_validate_clean_chain(tmp_path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    for i in range(5):
        chain.append({"submission_id": f"a{i}"})
    ok, errors = chain.validate()
    assert ok and errors == []


def test_validate_detects_mutation(tmp_path):
    path = tmp_path / "audit.jsonl"
    chain = AuditChain(path)
    chain.append({"submission_id": "a1"})
    chain.append({"submission_id": "a2"})

    # Tamper: modify the first entry
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["submission_id"] = "tampered"
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n")

    ok, errors = chain.validate()
    assert not ok
    assert any("_hash" in e for e in errors)


def test_validate_detects_prev_hash_break(tmp_path):
    path = tmp_path / "audit.jsonl"
    chain = AuditChain(path)
    chain.append({"submission_id": "a1"})
    chain.append({"submission_id": "a2"})
    chain.append({"submission_id": "a3"})

    # Delete middle entry → forward hash chain breaks
    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")

    ok, errors = chain.validate()
    assert not ok
    assert any("_prev_hash" in e for e in errors)


def test_len(tmp_path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    for i in range(3):
        chain.append({"submission_id": f"a{i}"})
    assert len(chain) == 3


# ---------------------------------------------------------------------------
# B4 — pipeline integration: every analyze() best-effort appends one entry
# ---------------------------------------------------------------------------

# Heavy fixture — text long enough to enter Phase 2 detectors and produce a
# realistic Determination. Kept minimal to keep the pipeline tests fast.
_SAMPLE_TEXT = (
    "Certainly! Here is a comprehensive overview of the topic you requested. "
    "First, it is important to understand the broader context. In conclusion, "
    "the synthesis presented above captures the key dimensions of the matter "
    "and provides a foundation for further exploration. Additionally, several "
    "considerations merit attention as we proceed through the analysis."
) * 3


def _audit_config(log_path: Path, *, enabled: bool = True) -> dict:
    """Minimal pipeline config that runs end-to-end with audit wired."""
    return {
        "_profile": "test_audit",
        "audit": {"enabled": enabled, "log_path": str(log_path)},
        "detectors": {
            "preamble": {"enabled": True, "weight": 1.0},
            "fingerprint_vocab": {"enabled": True, "weight": 0.8},
            "instruction_density": {"enabled": True, "weight": 0.75},
            "nssi": {"enabled": True, "weight": 0.7},
        },
        "cascade": {
            "phase1_short_circuit_high": 0.85,
            "phase1_short_circuit_low": 0.10,
            "phase2_short_circuit_high": 0.80,
            "phase2_short_circuit_low": 0.15,
        },
        "decision": {
            "red_threshold": 0.75,
            "amber_threshold": 0.50,
            "yellow_threshold": 0.25,
        },
    }


def test_pipeline_appends_audit_entry_when_enabled(tmp_path):
    log_path = tmp_path / "audit" / "chain.jsonl"
    pipeline = CairnPipeline(_audit_config(log_path, enabled=True))
    det = pipeline.analyze(_SAMPLE_TEXT)
    assert det is not None
    assert log_path.exists(), "audit log should be created on first append"

    chain = AuditChain(log_path)
    assert len(chain) == 1
    entries = chain._read_all()
    entry = entries[0]
    # Roadmap-mandated record shape:
    for field in (
        "submission_id",
        "text_hash",
        "profile",
        "p_llm",
        "determination",
        "calibration_status",
        "model_artifact_hash",
        "timestamp",
        "_prev_hash",
        "_hash",
    ):
        assert field in entry, f"missing audit field: {field}"
    assert entry["text_hash"].startswith("sha256:")
    assert entry["profile"] == "test_audit"
    assert entry["determination"] == det.label
    assert entry["calibration_status"] == "heuristic"
    assert entry["model_artifact_hash"] is None
    assert entry["_prev_hash"] == GENESIS


def test_pipeline_does_not_append_when_audit_disabled(tmp_path):
    log_path = tmp_path / "audit" / "chain.jsonl"
    cfg = _audit_config(log_path, enabled=False)
    pipeline = CairnPipeline(cfg)
    pipeline.analyze(_SAMPLE_TEXT)
    assert not log_path.exists(), (
        "audit log must not be touched when audit.enabled=false"
    )


def test_pipeline_audit_uses_caller_submission_id(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    pipeline = CairnPipeline(_audit_config(log_path))
    pipeline.analyze(_SAMPLE_TEXT, task_metadata={"submission_id": "call-42"})
    entry = AuditChain(log_path)._read_all()[0]
    assert entry["submission_id"] == "call-42"


def test_pipeline_audit_mints_submission_id_when_missing(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    pipeline = CairnPipeline(_audit_config(log_path))
    pipeline.analyze(_SAMPLE_TEXT)
    entry = AuditChain(log_path)._read_all()[0]
    sid = entry["submission_id"]
    assert isinstance(sid, str) and sid != ""


def test_audit_entry_never_contains_raw_text(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    pipeline = CairnPipeline(_audit_config(log_path))
    secret_marker = "ZZZ_unique_secret_canary_phrase_QQQ"
    text = _SAMPLE_TEXT + " " + secret_marker
    pipeline.analyze(text)
    raw = log_path.read_text(encoding="utf-8")
    assert secret_marker not in raw, (
        "audit chain must store hash only — raw submission text leaked"
    )


def test_failed_audit_append_does_not_fail_analyze(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.jsonl"
    pipeline = CairnPipeline(_audit_config(log_path))
    assert pipeline._audit_chain is not None

    def _boom(manifest):
        raise RuntimeError("simulated audit-store I/O failure")

    monkeypatch.setattr(pipeline._audit_chain, "append", _boom)
    # Must not raise — same best-effort discipline as drift / history.
    det = pipeline.analyze(_SAMPLE_TEXT)
    assert det is not None
    assert det.label in {
        "RED", "AMBER", "YELLOW", "GREEN", "UNCERTAIN", "MIXED", "SKIP",
    }


def test_audit_chain_validates_after_multiple_pipeline_calls(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    pipeline = CairnPipeline(_audit_config(log_path))
    for i in range(5):
        pipeline.analyze(_SAMPLE_TEXT, task_metadata={"submission_id": f"sub-{i}"})
    chain = AuditChain(log_path)
    assert len(chain) == 5
    ok, errors = chain.validate()
    assert ok, f"chain should validate clean; errors={errors}"


def test_pipeline_audit_prev_hash_links_explicitly(tmp_path):
    """validate() and append() could co-fail with matching bugs;
    assert the linkage directly so a regression in either side is
    caught even if validate() agrees with the bug."""
    log_path = tmp_path / "audit.jsonl"
    pipeline = CairnPipeline(_audit_config(log_path))
    for i in range(3):
        pipeline.analyze(_SAMPLE_TEXT, task_metadata={"submission_id": f"sub-{i}"})
    entries = AuditChain(log_path)._read_all()
    assert len(entries) == 3
    assert entries[0]["_prev_hash"] == GENESIS
    for i in range(1, len(entries)):
        assert entries[i]["_prev_hash"] == entries[i - 1]["_hash"], (
            f"entry {i} _prev_hash should equal entry {i - 1} _hash"
        )


def test_synthetic_1k_chain_validated_via_cli(tmp_path):
    """Roadmap test gate (verbatim): a 1k-call synthetic load produces a
    chain whose `cairn audit verify` returns valid=True and prev-hash
    links across all 1k entries. Exercised at the chain level (not full
    pipeline) since the gate proves chain integrity at scale, and
    full-pipeline iteration would dominate the suite runtime. Combined
    with a CLI-driven verify so a bug isolated to the dispatch path
    cannot pass silently."""
    log_path = tmp_path / "audit.jsonl"
    chain = AuditChain(log_path)
    for i in range(1000):
        chain.append({
            "submission_id": f"sub-{i:04d}",
            "text_hash": f"sha256:{i:064x}",
            "profile": "default",
            "p_llm": (i % 100) / 100.0,
            "determination": "AMBER" if i % 2 else "GREEN",
            "calibration_status": "heuristic",
            "model_artifact_hash": None,
            "timestamp": f"2026-05-07T00:00:{i % 60:02d}Z",
        })
    # Direct chain validation.
    ok, errors = chain.validate()
    assert ok and errors == []
    assert len(chain) == 1000
    # Spot-check a few links so a `validate()` bug that always returns
    # True cannot mask a broken chain.
    entries = chain._read_all()
    assert entries[0]["_prev_hash"] == GENESIS
    for idx in (1, 500, 999):
        assert entries[idx]["_prev_hash"] == entries[idx - 1]["_hash"]
    # CLI-driven verification of the same chain.
    runner = CliRunner()
    result = runner.invoke(cairn_cli, ["audit", "verify", "--log-path", str(log_path)])
    assert result.exit_code == 0, result.output
    assert "valid=True" in result.output
    assert "entries=1000" in result.output


# ---------------------------------------------------------------------------
# B4 — `cairn audit verify` CLI subcommand
# ---------------------------------------------------------------------------

def _seed_chain(path: Path, n: int = 3) -> None:
    chain = AuditChain(path)
    for i in range(n):
        chain.append({
            "submission_id": f"cli-{i}",
            "text_hash": f"sha256:{i:064x}",
            "profile": "default",
            "p_llm": 0.5,
            "determination": "YELLOW",
            "calibration_status": "heuristic",
            "model_artifact_hash": None,
            "timestamp": f"2026-05-07T00:00:0{i}Z",
        })


def test_cli_audit_verify_reports_valid_chain(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    _seed_chain(log_path, n=3)
    runner = CliRunner()
    result = runner.invoke(cairn_cli, ["audit", "verify", "--log-path", str(log_path)])
    assert result.exit_code == 0, result.output
    assert "valid=True" in result.output
    assert "entries=3" in result.output


def test_cli_audit_verify_detects_tampering(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    _seed_chain(log_path, n=3)
    # Tamper with the first entry's payload — recomputed hash will mismatch.
    lines = log_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["determination"] = "RED"
    lines[0] = json.dumps(first)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cairn_cli, ["audit", "verify", "--log-path", str(log_path)])
    assert result.exit_code != 0, result.output
    assert "valid=False" in result.output


def test_cli_audit_verify_handles_missing_log(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    runner = CliRunner()
    result = runner.invoke(cairn_cli, ["audit", "verify", "--log-path", str(missing)])
    # An empty/missing chain is vacuously valid; entries=0.
    assert result.exit_code == 0, result.output
    assert "entries=0" in result.output


# ---------------------------------------------------------------------------
# D1 — audit chain records overlay's `_profile` after `_deep_merge`
# ---------------------------------------------------------------------------
# Pre-fix `_deep_merge` stripped every `_`-prefixed key from the override,
# so an extending profile (`_profile: overlay`, `_extends: default`) merged
# into a base produced a config that still reported `_profile: default` —
# and the audit chain logged the wrong profile name on every analyze call.

def test_pipeline_audit_records_extending_profile_name(tmp_path):
    from cairn.config import _deep_merge
    log_path = tmp_path / "audit.jsonl"
    base = _audit_config(log_path, enabled=True)
    overlay = {
        "_profile": "profile_master_minus_arxiv",
        "audit": {"enabled": True, "log_path": str(log_path)},
    }
    merged = _deep_merge(base, overlay)
    assert merged["_profile"] == "profile_master_minus_arxiv"

    pipeline = CairnPipeline(merged)
    pipeline.analyze(_SAMPLE_TEXT)
    entry = AuditChain(log_path)._read_all()[0]
    assert entry["profile"] == "profile_master_minus_arxiv", (
        f"audit chain logged the base's profile instead of the overlay's "
        f"after _deep_merge: got {entry['profile']!r}"
    )
