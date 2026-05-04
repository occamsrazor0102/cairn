"""CAIRN evaluation module — dataset loading, metrics, runner, ablation."""
from cairn.evaluation.dataset import (
    EvalSample,
    load_dataset,
    save_dataset,
    build_dataset,
)
from cairn.evaluation.metrics import (
    auroc,
    ece,
    brier,
    tpr_at_fpr,
    confusion_at_threshold,
    per_tier_breakdown,
    per_attack_breakdown,
    summarize,
)
from cairn.evaluation.runner import EvalReport, eval_report_to_dict, run_eval
from cairn.evaluation.ablation import AblationReport, run_ablation, verdict_for
from cairn.evaluation.robustness import RobustnessReport, run_robustness_eval
from cairn.evaluation.fairness import FairnessReport, run_fairness_eval

__all__ = [
    "EvalSample",
    "load_dataset",
    "save_dataset",
    "build_dataset",
    "auroc",
    "ece",
    "brier",
    "tpr_at_fpr",
    "confusion_at_threshold",
    "per_tier_breakdown",
    "per_attack_breakdown",
    "summarize",
    "EvalReport",
    "eval_report_to_dict",
    "run_eval",
    "AblationReport",
    "run_ablation",
    "verdict_for",
    "RobustnessReport",
    "run_robustness_eval",
    "FairnessReport",
    "run_fairness_eval",
]
