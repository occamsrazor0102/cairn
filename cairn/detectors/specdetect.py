"""SpecDetect -- frequency-domain features of per-token log-probabilities.

Luo et al., "SpecDetect: Simple, Fast, and Training-Free Detection of
LLM-Generated Text via Spectral Analysis", arXiv 2508.11343 (AAAI 26).

Theory: human-written text shows higher variability in per-token
log-probability under a fixed proxy LM than LLM-generated text. The
paper frames this as "DFT total energy" of the zero-mean log-prob
signal. By Parseval's identity, that total scalar is mathematically
equivalent to `n * variance(log-probs)` -- redundant with the
`surprisal_std` feature CAIRN already exposes via `surprisal_dynamics`.

The novel value is in how the variance is *distributed across
frequencies*, not the total. This detector exposes:

- `dft_total_energy`           -- scalar, for paper-comparison
- `dft_low_band_energy`        -- fraction of energy in bottom 25% bins
- `dft_mid_band_energy`        -- fraction in middle 50% bins
- `dft_high_band_energy`       -- fraction in top 25% bins
- `dft_low_mid_ratio`          -- low/mid ratio (slow vs jittery shape)
- `dft_spectral_entropy`       -- Shannon entropy over normalized spectrum

Two texts with identical surprisal_std can have very different spectral
shapes; that's the EBM gain over `surprisal_dynamics` alone.

PRIVACY: proxy LM runs locally -- no text leaves the process.
"""
from __future__ import annotations

import math
from typing import Any

from cairn.contracts import LayerResult
from cairn.detectors._utils import interpolate as _interpolate

try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Lower total energy => more LLM-like (LLMs sample near the mode, so
# per-token surprisal is more uniform). Anchors are normalized-energy
# units (sum |X_k|^2 / n^2) -- hand-picked, NOT calibrated. Replaced by
# isotonic calibration once a fitter pass runs.
_HEURISTIC_TOTAL_ENERGY_TO_P_LLM = [
    (0.5, 0.92),
    (1.0, 0.78),
    (1.5, 0.55),
    (2.0, 0.30),
    (3.0, 0.10),
    (5.0, 0.05),
]

_MIN_TOKENS_FOR_FFT = 16


class SpecDetectDetector:
    id = "specdetect"
    domain = "prose"
    compute_cost = "moderate"

    def __init__(self):
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded_key: tuple[str, str] | None = None

    # ----- model loading -----

    def _ensure_loaded(self, model_name: str, device: str) -> None:
        key = (model_name, device)
        if self._model is not None and self._loaded_key == key:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # fp16 on GPU keeps the proxy LM small enough to fit T4 (16 GB).
        # GPT-J-6B at fp32 is ~24 GB and OOMs; at fp16 it's ~12 GB. CPU
        # path stays fp32 since most CPUs don't accelerate fp16.
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self._model.eval()
        self._loaded_key = key

    # ----- log-prob scoring -----

    def _per_token_log_probs(
        self, input_ids: Any, device: str
    ) -> list[float]:
        """Return log P(x_{t+1} | x_<=t) for t = 0..T-2 as a python list."""
        with torch.no_grad():
            out = self._model(input_ids.to(device))
        logits = out.logits[0, :-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = input_ids[0, 1:].to(log_probs.device)
        token_lp = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        return [float(lp.item()) for lp in token_lp]

    # ----- spectral features -----

    @staticmethod
    def _spectral_features(log_probs: list[float]) -> dict[str, float]:
        """DFT-derived features. Returns total energy + band fractions + ratio + entropy.

        Pure-Python (numpy only) so this is unit-testable without a GPU.
        Energy is normalized by n^2 to keep magnitudes comparable across
        text lengths.
        """
        import numpy as np

        n = len(log_probs)
        signal = np.asarray(log_probs, dtype=np.float64)
        signal = signal - signal.mean()                     # zero-mean

        spectrum = np.fft.fft(signal)
        power = np.abs(spectrum) ** 2                        # |X_k|^2
        total_energy = float(power.sum()) / (n * n)          # normalized

        # Only consider the non-redundant half of the spectrum (FFT is
        # symmetric for real input). Drop the DC bin (k=0) -- it's zero
        # after the zero-mean step anyway, but excluding it explicitly
        # keeps the band fractions well-defined.
        half = n // 2
        power_half = power[1:half + 1]                       # skip DC
        total_half = float(power_half.sum())
        if total_half < 1e-12:
            # Degenerate case: signal is essentially constant. All bands
            # zero; assign uniform fractions so the row stays usable.
            return {
                "dft_total_energy": total_energy,
                "dft_low_band_energy": 1.0 / 3,
                "dft_mid_band_energy": 1.0 / 3,
                "dft_high_band_energy": 1.0 / 3,
                "dft_low_mid_ratio": 1.0,
                "dft_spectral_entropy": math.log(max(len(power_half), 1)),
            }

        m = len(power_half)
        low_end = max(1, m // 4)
        mid_end = max(low_end + 1, 3 * m // 4)
        low = float(power_half[:low_end].sum()) / total_half
        mid = float(power_half[low_end:mid_end].sum()) / total_half
        high = float(power_half[mid_end:].sum()) / total_half
        low_mid_ratio = low / max(mid, 1e-9)

        # Spectral entropy: Shannon entropy of the normalized power
        # distribution. Higher = flatter spectrum (white-noise-like).
        normalized = power_half / total_half
        entropy = -float(np.sum(normalized * np.log(normalized + 1e-12)))

        return {
            "dft_total_energy": total_energy,
            "dft_low_band_energy": low,
            "dft_mid_band_energy": mid,
            "dft_high_band_energy": high,
            "dft_low_mid_ratio": low_mid_ratio,
            "dft_spectral_entropy": entropy,
        }

    def _resolve_device(self) -> str:
        """Pick 'cuda' when a real torch with CUDA is present, else 'cpu'.

        Guarded against NameError: `_HAS_TORCH` may be forced True (tests
        stub the scorer) or torch may have failed to import at module load,
        in which case the module-global `torch` symbol doesn't exist.
        """
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except NameError:
            return "cpu"

    # ----- main entry point -----

    def analyze(self, text: str, config: dict) -> LayerResult:
        n_words = len(text.split())
        if n_words < 50:
            return _skip(self, {"skipped": "too_short", "n_words": n_words})
        if not _HAS_TORCH:
            return _skip(self, {
                "skipped": "torch_unavailable",
                "hint": "pip install cairn[tier2] to enable this detector",
            })

        model_name = config.get("model", "gpt2")
        max_tokens = int(config.get("max_tokens", 512))
        device = config.get("device") or self._resolve_device()

        try:
            self._ensure_loaded(model_name, device)
        except Exception as e:
            return _skip(self, {
                "skipped": "model_load_failed",
                "error": f"{type(e).__name__}: {e}",
                "model": model_name,
            })

        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_tokens
        )
        input_ids = enc["input_ids"]
        n_tokens = int(input_ids.shape[1])
        truncated = n_tokens >= max_tokens

        if n_tokens < _MIN_TOKENS_FOR_FFT:
            return _skip(self, {
                "skipped": "too_few_tokens_for_fft",
                "n_tokens": n_tokens,
                "min_required": _MIN_TOKENS_FOR_FFT,
            })

        try:
            log_probs = self._per_token_log_probs(input_ids, device)
        except Exception as e:
            return _skip(self, {
                "skipped": "scoring_failed",
                "error": f"{type(e).__name__}: {e}",
            })

        feats = self._spectral_features(log_probs)
        p_llm = _interpolate(feats["dft_total_energy"], _HEURISTIC_TOTAL_ENERGY_TO_P_LLM)

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
            raw_score=float(feats["dft_total_energy"]),
            p_llm=p_llm,
            confidence=min(0.4 + n_words / 800, 0.85),
            signals={
                "n_tokens": n_tokens,
                "dft_total_energy": round(feats["dft_total_energy"], 6),
                "dft_low_band_energy": round(feats["dft_low_band_energy"], 4),
                "dft_mid_band_energy": round(feats["dft_mid_band_energy"], 4),
                "dft_high_band_energy": round(feats["dft_high_band_energy"], 4),
                "dft_low_mid_ratio": round(feats["dft_low_mid_ratio"], 4),
                "dft_spectral_entropy": round(feats["dft_spectral_entropy"], 4),
                "model": model_name,
                "device": device,
                "truncated": truncated,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2"],
            compute_cost=self.compute_cost,
            min_text_length=50,
        )

    def calibrate(self, labeled_data: list) -> None:
        pass


def _skip(detector, signals: dict) -> LayerResult:
    return LayerResult(
        layer_id=detector.id, domain=detector.domain,
        raw_score=0.0, p_llm=0.5, confidence=0.0,
        signals=signals,
        determination="SKIP",
        attacker_tiers=["A0", "A1", "A2"],
        compute_cost=detector.compute_cost,
        min_text_length=50,
    )


DETECTOR = SpecDetectDetector()
