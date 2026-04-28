"""CAIRN adversarial testing — generate evasion samples for stress-testing."""
from cairn.adversarial.registry import Attack, register, get_attack, list_attacks
from cairn.adversarial.generator import generate
from cairn.adversarial import transforms as _transforms  # noqa: F401 — registers attacks

__all__ = ["Attack", "register", "get_attack", "list_attacks", "generate"]
