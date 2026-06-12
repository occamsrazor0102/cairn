# cairn/detectors/surprisal_dynamics.py
import math
from cairn.contracts import LayerResult

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _get_token_surprisals(text: str, model, tokenizer, device: str) -> list[float]:
    """Returns per-token surprisal (negative log-prob) values."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # Get per-token log-probs
        logits = outputs.logits[0, :-1]  # shift
        targets = inputs["input_ids"][0, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = log_probs[range(len(targets)), targets]
        return [-lp.item() for lp in token_lp]


def _windowed_variance(values: list[float], window_size: int) -> list[float]:
    """Returns variance in each sliding window."""
    if len(values) < window_size:
        return []
    variances = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        mean = sum(window) / window_size
        var = sum((x - mean) ** 2 for x in window) / window_size
        variances.append(var)
    return variances


def _moments(xs: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, skewness, excess-kurtosis) of ``xs`` in pure Python."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(var)
    if std < 1e-9:
        return mean, 0.0, 0.0, 0.0
    m3 = sum((x - mean) ** 3 for x in xs) / n
    m4 = sum((x - mean) ** 4 for x in xs) / n
    skew = m3 / (std ** 3)
    kurt = m4 / (std ** 4) - 3.0
    return mean, std, skew, kurt


def _lag1_autocorr(xs: list[float]) -> float:
    """Lag-1 autocorrelation of ``xs`` (0.0 when undefined)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    denom = sum((x - mean) ** 2 for x in xs)
    if denom < 1e-9:
        return 0.0
    num = sum((xs[i] - mean) * (xs[i + 1] - mean) for i in range(n - 1))
    return num / denom


def _diveye_features(surprisals: list[float]) -> dict[str, float]:
    """DivEye-style diversity features over the surprisal series.

    Reimplements the feature *definitions* from DivEye (Basani & Chen,
    arXiv:2509.18880) — 9 statistics capturing the distribution and the
    first/second-order temporal fluctuation of per-token surprisal, which
    is what makes the method robust to paraphrase and OOD where a few
    hand-weighted scalars invert. (The DivEye reference code is CC-BY-NC-SA;
    these are clean-room reimplementations of the published statistics, not
    vendored code.)

    Returns the 9 features keyed for the EBM assembler. The fitted EBM
    learns the mapping; we emit no heuristic p(LLM) from them.
    """
    s_mean, s_std, s_skew, s_kurt = _moments(surprisals)
    d1 = [surprisals[i + 1] - surprisals[i] for i in range(len(surprisals) - 1)]
    d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
    _, d1_std, _, _ = _moments(d1)
    d1_absmean = sum(abs(x) for x in d1) / len(d1) if d1 else 0.0
    d1_autocorr = _lag1_autocorr(d1)
    _, d2_std, _, _ = _moments(d2)
    d2_absmean = sum(abs(x) for x in d2) / len(d2) if d2 else 0.0
    return {
        "surprisal_skew": round(s_skew, 4),
        "surprisal_kurtosis": round(s_kurt, 4),
        "surprisal_d1_std": round(d1_std, 4),
        "surprisal_d1_absmean": round(d1_absmean, 4),
        "surprisal_d1_autocorr": round(d1_autocorr, 4),
        "surprisal_d2_std": round(d2_std, 4),
        "surprisal_d2_absmean": round(d2_absmean, 4),
        # s_mean / s_std are already surfaced as surprisal_mean/std below;
        # included here as the 8th/9th DivEye moments for completeness of
        # the distributional block.
        "surprisal_moment_mean": round(s_mean, 4),
        "surprisal_moment_std": round(s_std, 4),
    }


class SurprisalDynamicsDetector:
    id = "surprisal_dynamics"
    domain = "prose"
    compute_cost = "moderate"

    def __init__(self, model_name: str = "gpt2"):
        if not _HAS_TORCH:
            raise ImportError("torch/transformers required. Run: pip install 'cairn[tier2]'")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def analyze(self, text: str, config: dict) -> LayerResult:
        surprisals = _get_token_surprisals(text, self._model, self._tokenizer, self._device)
        if len(surprisals) < 20:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0, signals={"skipped": "insufficient_tokens"},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2", "A3"],
                compute_cost=self.compute_cost, min_text_length=150,
            )

        n = len(surprisals)
        third = n // 3
        early = surprisals[:third]
        late = surprisals[2 * third:]

        # Key signals
        surprisal_mean = sum(surprisals) / n
        surprisal_std = math.sqrt(sum((x - surprisal_mean) ** 2 for x in surprisals) / n)
        early_var = sum((x - sum(early) / len(early)) ** 2 for x in early) / max(len(early), 1)
        late_var = sum((x - sum(late) / len(late)) ** 2 for x in late) / max(len(late), 1)
        late_volatility_ratio = late_var / max(early_var, 0.001)

        # Surprisal diversity: entropy of binned distribution
        bins = [0] * 20
        for s in surprisals:
            bin_idx = min(int(s / 1.5), 19)
            bins[bin_idx] += 1
        total = sum(bins)
        entropy = -sum((b / total) * math.log(b / total + 1e-9) for b in bins if b > 0)
        surprisal_diversity = entropy / math.log(20)  # normalize to 0–1

        # Windowed trajectory slope
        win_vars = _windowed_variance(surprisals, window_size=20)
        if len(win_vars) > 2:
            n_w = len(win_vars)
            x_mean = (n_w - 1) / 2
            slope = sum((i - x_mean) * (v - sum(win_vars) / n_w) for i, v in enumerate(win_vars))
            slope /= max(sum((i - x_mean) ** 2 for i in range(n_w)), 1e-9)
        else:
            slope = 0.0

        # Kurtosis excess
        mu = surprisal_mean
        m4 = sum((x - mu) ** 4 for x in surprisals) / n
        kurtosis_excess = m4 / max(surprisal_std ** 4, 1e-9) - 3

        # The previous hand-weighted p(LLM) — built from late_volatility,
        # surprisal_diversity and slope — was the component that *inverted*
        # out-of-domain (AUROC 0.27–0.36 on M4/RAID/MAGE vs 0.59 in-domain;
        # see docs/benchmarks/calib2000/per_detector_auroc.csv). We no longer
        # emit that point estimate. Instead we expose the richer DivEye-style
        # feature block and suppress the heuristic verdict, letting the EBM
        # learn the correct (and domain-aware) mapping at calibration time.
        #
        # IMPORTANT: the suppressed result uses determination="UNCERTAIN"
        # (NOT "SKIP"). FeatureAssembler.assemble() drops SKIP results from
        # by_id, which would silently discard every surprisal_* feature
        # before the EBM ever sees it. UNCERTAIN keeps the row in the feature
        # vector; confidence=0.0 keeps it out of the naive aggregate and the
        # cascade short-circuit (both filter on confidence > 0), so a broken
        # heuristic can't move the verdict while its features still feed
        # fusion. A loaded per-detector isotonic calibrator overrides p_llm
        # from raw_score in cairn/pipeline.py at read time.
        diveye = _diveye_features(surprisals)

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=late_volatility_ratio,
            p_llm=0.5,
            confidence=0.0,
            signals={
                "surprisal_mean": round(surprisal_mean, 3),
                "surprisal_std": round(surprisal_std, 3),
                "late_volatility_ratio": round(late_volatility_ratio, 3),
                "surprisal_diversity": round(surprisal_diversity, 3),
                "window_trajectory_slope": round(slope, 5),
                "kurtosis_excess": round(kurtosis_excess, 3),
                "n_tokens": n,
                **diveye,
                "reason": "heuristic_suppress",
                "notes": (
                    "heuristic point estimate suppressed (inverted OOD); "
                    "DivEye-style features feed the EBM pending calibrator retrain"
                ),
            },
            determination="UNCERTAIN",
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=150,
        )

    def calibrate(self, labeled_data: list) -> None: pass


if _HAS_TORCH:
    # Lazy instantiation: don't load model at import time
    class _LazyDetector:
        id = "surprisal_dynamics"
        domain = "prose"
        compute_cost = "moderate"
        _instance = None

        def analyze(self, text: str, config: dict) -> LayerResult:
            if self._instance is None:
                model_name = config.get("model_name", "gpt2")
                self._instance = SurprisalDynamicsDetector(model_name)
            return self._instance.analyze(text, config)

        def calibrate(self, labeled_data: list) -> None: pass

    DETECTOR = _LazyDetector()
