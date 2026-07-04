"""LAPD -- Log-likelihood Alignment Preference Discrepancy.

Wu et al., "Alignment Imprint: Zero-Shot AI-Generated Text Detection via
Provable Preference Discrepancy", arXiv 2604.16923.

Theory: a base LM theta and its aligned (instruction-tuned) descendant
phi disagree on per-token probabilities. The disagreement -- Delta_t =
log P_phi - log P_theta -- is small on human-written tokens and large
on LLM-generated tokens, because aligned models prefer the same
continuations that other aligned models produced. Weighting Delta_t by
the aligned-model self-information I_phi(x_t) = -log P_phi(x_t)
de-noises high-entropy positions where Delta_t is unreliable. The raw
aggregate S(x) = sum_t I_phi(x_t) * Delta_t is standardized via
perturbation sampling to S_LAPD(x) = (S - mu_tilde) / sigma_tilde.

Implementation notes:
  * Requires TWO models that share a tokenizer: an aligned (instruct)
    LM and the corresponding base LM. Default pair is the Falcon-7B
    family (Apache-2.0, no Meta gate).
  * 4-bit quantization is the default to fit the pair on a single T4.
  * Perturbation sampling replaces a fraction of tokens with random
    tokens drawn from the input itself, then recomputes S. n=50
    perturbations is the production default; the spike measures
    smaller budgets.
  * SKIP when transformers/torch are missing, when the two models'
    tokenizers disagree, or when the perturbation variance collapses.

PRIVACY: both models run locally -- no text leaves the process.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from cairn.contracts import LayerResult
from cairn.detectors._utils import interpolate as _interpolate

try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import bitsandbytes  # type: ignore  # noqa: F401
    _HAS_BITSANDBYTES = True
except ImportError:
    _HAS_BITSANDBYTES = False

# S_LAPD is roughly N(0, 1) under human authorship and shifted positive
# under LLM authorship per the paper's distribution plots. Hand-picked
# anchors -- NOT calibrated to a labeled dataset. Replaced by isotonic
# calibration once the calibrator pass runs on the cached features.
_HEURISTIC_S_LAPD_TO_P_LLM = [
    (-2.0, 0.02), (-1.0, 0.05), (0.0, 0.10),
    (0.5, 0.30), (1.0, 0.55),
    (2.0, 0.78), (3.0, 0.90), (5.0, 0.97),
]

# Below this variance the standardization is numerically unstable;
# abstain rather than emit a pegged value driven by noise. Mirrors the
# same floor used by `cairn/detectors/perturbation.py`.
_MIN_PERTURBATION_VARIANCE = 1e-4


class LAPDDetector:
    id = "lapd"
    domain = "universal"
    compute_cost = "expensive"

    def __init__(self):
        self._aligned_model: Any = None
        self._base_model: Any = None
        self._aligned_tokenizer: Any = None
        self._base_tokenizer: Any = None
        self._loaded_key: tuple[str, str, str, bool] | None = None

    # ----- model loading -----

    def _load_one(self, name: str, device: str, quantize_4bit: bool) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(name)
        kwargs: dict[str, Any] = {}
        if quantize_4bit and _HAS_BITSANDBYTES:
            # transformers >= 4.40 deprecated `load_in_4bit=True` as a direct
            # from_pretrained kwarg; the supported path is a BitsAndBytesConfig
            # passed as `quantization_config`. Imported here (not at module
            # scope) because the symbol only exists when bitsandbytes is
            # available, and we already skipped this branch when it isn't.
            from transformers import BitsAndBytesConfig  # type: ignore
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                # fp16 compute keeps activations in 2-byte tensors; default
                # fp32 doubles activation memory and crowds out the second
                # model on smaller GPUs.
                bnb_4bit_compute_dtype=torch.float16,
            )
            # Pin to GPU 0 explicitly. `device_map="auto"` is conservative
            # about leaving headroom when a second model is already resident,
            # and bitsandbytes 4-bit refuses CPU offload. Force everything on
            # the GPU so the second model's load doesn't dispatch some layers
            # to CPU/disk (the validation error we hit before this comment).
            kwargs["device_map"] = {"": 0}
            # Load weights as fp16 (not fp32) and avoid the transformers
            # warmup allocator -- both halve peak memory during the load
            # window so two 7B models can fit on a 24 GB GPU.
            kwargs["torch_dtype"] = torch.float16
            kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if not (quantize_4bit and _HAS_BITSANDBYTES):
            model = model.to(device)
        model.eval()
        return model, tokenizer

    def _ensure_loaded(
        self, aligned_name: str, base_name: str, device: str, quantize_4bit: bool
    ) -> None:
        key = (aligned_name, base_name, device, bool(quantize_4bit))
        if self._aligned_model is not None and self._loaded_key == key:
            return
        self._aligned_model, self._aligned_tokenizer = self._load_one(
            aligned_name, device, quantize_4bit
        )
        # Free the warmup allocator's temporary fp16 buffers before the second
        # model's load. Without this, the second from_pretrained sees a
        # half-empty GPU but the cached unallocated reservation prevents new
        # large contiguous allocations.
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._base_model, self._base_tokenizer = self._load_one(
            base_name, device, quantize_4bit
        )
        self._loaded_key = key

    @staticmethod
    def _tokenizer_fingerprint(tokenizer: Any) -> str:
        """Stable identifier for a tokenizer's vocab.

        Used to detect aligned/base pairs that don't share a tokenizer
        -- their per-token log-probs aren't comparable and Delta_t
        would be meaningless. Hashes the first 1024 entries of the
        vocab to keep this O(1) at load time.
        """
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            return "unknown"
        pairs = sorted(vocab.items(), key=lambda kv: kv[1])[:1024]
        digest = hashlib.sha256()
        for tok, idx in pairs:
            digest.update(f"{idx}\t{tok}\n".encode("utf-8", errors="replace"))
        return digest.hexdigest()[:16]

    # ----- log-prob scoring -----

    def _per_token_log_probs(
        self, model: Any, tokenizer: Any, input_ids: Any, device: str
    ) -> Any:
        """Return per-token log-probs of `input_ids` under `model`.

        Output shape: (T-1,) -- log P(x_{t+1} | x_<=t) for t = 0..T-2.
        """
        with torch.no_grad():
            out = model(input_ids.to(device))
        logits = out.logits[0, :-1, :]                # (T-1, V)
        log_probs = torch.log_softmax(logits, dim=-1) # (T-1, V)
        targets = input_ids[0, 1:]                    # (T-1,)
        return log_probs.gather(1, targets.unsqueeze(-1).to(log_probs.device)).squeeze(-1)

    def _score(self, input_ids: Any, device: str) -> float:
        """Compute S(x) = sum_t I_phi(x_t) * Delta(x_t) for input_ids."""
        lp_phi = self._per_token_log_probs(
            self._aligned_model, self._aligned_tokenizer, input_ids, device
        )
        lp_theta = self._per_token_log_probs(
            self._base_model, self._base_tokenizer, input_ids, device
        )
        delta = lp_phi - lp_theta                     # (T-1,)
        i_phi = -lp_phi                               # (T-1,)
        return float((i_phi * delta).sum().item())

    # ----- perturbation -----

    @staticmethod
    def _perturb_ids(input_ids: Any, rate: float, rng: random.Random) -> Any:
        """Replace `rate` fraction of token IDs with others drawn from the same sequence."""
        ids = input_ids[0].tolist()
        n = len(ids)
        if n < 4:
            return input_ids
        n_swaps = max(1, int(n * rate))
        positions = rng.sample(range(n), min(n_swaps, n))
        for pos in positions:
            ids[pos] = rng.choice(ids)
        return input_ids.new_tensor([ids])

    # ----- main entry point -----

    def analyze(self, text: str, config: dict) -> LayerResult:
        n_words = len(text.split())
        if n_words < 100:
            return _skip(self, {"skipped": "too_short", "n_words": n_words})
        if not _HAS_TORCH:
            return _skip(self, {
                "skipped": "torch_unavailable",
                "hint": "pip install cairn[tier2] to enable this detector",
            })

        aligned_name = config.get("aligned_model", "tiiuae/falcon-7b-instruct")
        base_name = config.get("base_model", "tiiuae/falcon-7b")
        quantize_4bit = bool(config.get("quantize_4bit", True))
        n_perturbations = int(config.get("n_perturbations", 50))
        perturbation_rate = float(config.get("perturbation_rate", 0.10))
        max_tokens = int(config.get("max_tokens", 512))
        seed = int(config.get("seed", 42))
        device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        if quantize_4bit and not _HAS_BITSANDBYTES:
            return _skip(self, {
                "skipped": "bitsandbytes_unavailable",
                "hint": "pip install cairn[tier3] for 4-bit; "
                        "or set detectors.lapd.quantize_4bit: false",
            })

        try:
            self._ensure_loaded(aligned_name, base_name, device, quantize_4bit)
        except Exception as e:
            return _skip(self, {
                "skipped": "model_load_failed",
                "error": f"{type(e).__name__}: {e}",
                "aligned_model": aligned_name,
                "base_model": base_name,
            })

        aligned_fp = self._tokenizer_fingerprint(self._aligned_tokenizer)
        base_fp = self._tokenizer_fingerprint(self._base_tokenizer)
        if aligned_fp != base_fp:
            return _skip(self, {
                "skipped": "tokenizer_mismatch",
                "aligned_fingerprint": aligned_fp,
                "base_fingerprint": base_fp,
            })

        enc = self._aligned_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_tokens
        )
        input_ids = enc["input_ids"]
        n_tokens = int(input_ids.shape[1])
        truncated = n_tokens >= max_tokens

        try:
            raw_s = self._score(input_ids, device)
            rng = random.Random(seed)
            perturbed_scores: list[float] = []
            for _ in range(n_perturbations):
                pert_ids = self._perturb_ids(input_ids, perturbation_rate, rng)
                perturbed_scores.append(self._score(pert_ids, device))
        except Exception as e:
            return _skip(self, {
                "skipped": "scoring_failed",
                "error": f"{type(e).__name__}: {e}",
            })

        mu_tilde = sum(perturbed_scores) / len(perturbed_scores)
        var_tilde = sum((s - mu_tilde) ** 2 for s in perturbed_scores) / len(perturbed_scores)
        if var_tilde < _MIN_PERTURBATION_VARIANCE:
            return _skip(self, {
                "skipped": "insufficient_perturbation_variance",
                "var_tilde": var_tilde,
                "n_perturbations": n_perturbations,
                "hint": "perturbation samples clustered too tightly; "
                        "try a longer text or a higher perturbation_rate",
            })
        sigma_tilde = math.sqrt(var_tilde)
        s_lapd = (raw_s - mu_tilde) / sigma_tilde
        p_llm = _interpolate(s_lapd, _HEURISTIC_S_LAPD_TO_P_LLM)

        if p_llm >= 0.75:
            determination = "RED"
        elif p_llm >= 0.50:
            determination = "AMBER"
        elif p_llm >= 0.25:
            determination = "YELLOW"
        else:
            determination = "GREEN"

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=float(s_lapd),
            p_llm=p_llm,
            confidence=min(0.4 + n_words / 1000, 0.85),
            signals={
                "raw_s": round(raw_s, 4),
                "s_lapd": round(s_lapd, 4),
                "mu_tilde": round(mu_tilde, 4),
                "sigma_tilde": round(sigma_tilde, 4),
                "n_tokens": n_tokens,
                "n_perturbations": n_perturbations,
                "model_pair": f"{aligned_name} + {base_name}",
                "quantized": bool(quantize_4bit and _HAS_BITSANDBYTES),
                "device": device,
                "truncated": truncated,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=100,
        )

    def calibrate(self, labeled_data: list) -> None:
        pass


def _skip(detector, signals: dict) -> LayerResult:
    return LayerResult(
        layer_id=detector.id, domain=detector.domain,
        raw_score=0.0, p_llm=0.5, confidence=0.0,
        signals=signals,
        determination="SKIP",
        attacker_tiers=["A0", "A1", "A2", "A3"],
        compute_cost=detector.compute_cost,
        min_text_length=100,
    )


DETECTOR = LAPDDetector()
