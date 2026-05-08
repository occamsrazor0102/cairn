# cairn/config.py
import logging
import re
from pathlib import Path
import yaml
import copy

logger = logging.getLogger("cairn.config")

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
PATTERNS_DIR = CONFIGS_DIR / "patterns"

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

REQUIRED_KEYS = {"detectors", "cascade", "decision"}

class ConfigError(Exception):
    pass

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

# Identity metadata that an overlay declares as its own. These survive
# the merge so an extending profile's name wins over its base — otherwise
# every extending profile silently reports its base's name in /health,
# audit-chain manifests, and any other surface that reads `_profile`.
# Other underscore-prefixed keys (`_extends`) are one-shot directives
# whose only role is to trigger expansion in `load_config`; they must
# not cascade through inheritance chains.
_PRESERVED_META_KEYS = frozenset({"_profile", "_description"})


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursively for nested dicts.

    Most underscore-prefixed keys are skipped so meta-directives like
    `_extends` don't cascade. Identity metadata listed in
    `_PRESERVED_META_KEYS` is kept from the override.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key.startswith("_") and key not in _PRESERVED_META_KEYS:
            continue  # skip meta-directives like `_extends`
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result

def load_config(path: Path | str) -> dict:
    path = Path(path)
    raw = _load_yaml(path)

    # Handle _extends
    if "_extends" in raw:
        base_name = raw["_extends"]
        base_path = CONFIGS_DIR / f"{base_name}.yaml"
        base = load_config(base_path)
        cfg = _deep_merge(base, raw)
    else:
        cfg = raw

    missing = REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise ConfigError(f"Config missing required keys: {missing}")

    return cfg

def resolve_profile_path(name: str) -> Path:
    """Return the absolute path to a profile's YAML in CONFIGS_DIR.

    Rejects names with path separators, `..`, or characters outside the
    `[A-Za-z0-9_-]` set so a caller can't escape the configs directory.
    """
    if not isinstance(name, str) or not _PROFILE_NAME_RE.fullmatch(name):
        raise ConfigError(f"invalid profile name: {name!r}")
    return CONFIGS_DIR / f"{name}.yaml"


def list_profiles() -> list[dict]:
    """List available profiles in CONFIGS_DIR.

    Skips files in subdirectories (e.g. patterns/) and files starting
    with '_'. For each profile, returns its canonical name (from
    `_profile` field if present, else filename stem), path, and optional
    description/extends metadata.
    """
    profiles: list[dict] = []
    for p in sorted(CONFIGS_DIR.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        try:
            raw = _load_yaml(p)
        except Exception:
            logger.warning("Skipped unparseable profile %s", p.name, exc_info=True)
            continue
        name = raw.get("_profile") or p.stem
        profiles.append({
            "name": str(name),
            "path": str(p),
            "extends": raw.get("_extends"),
            "description": raw.get("_description"),
        })
    return profiles


def get_pattern_list(name: str) -> list | dict:
    """Load a pattern file from configs/patterns/."""
    path = PATTERNS_DIR / f"{name}.yaml"
    data = _load_yaml(path)
    # Return the most useful top-level structure
    if "words" in data:
        return data["words"]
    return data
