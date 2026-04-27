"""Tiny `.env` loader, no dependency on python-dotenv.

Used by scripts that need provider API keys (Anthropic, OpenAI,
Gemini) and/or HuggingFace tokens (HF_TOKEN). Call `load_dotenv()`
at the top of `main()` before any code that reads env vars.

Existing env vars take precedence — explicit `export` always wins
over `.env`. No-op if `.env` doesn't exist.

Format:
    # comments and blank lines OK
    KEY=value
    KEY="value with spaces"
    KEY='value'
"""
from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    """Repo root. We're at beet/dotenv.py so two levels up."""
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | str | None = None) -> None:
    """Load `.env` into os.environ. No-op if file is missing."""
    env_path = Path(path) if path else _project_root() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val
