"""Compare robust_v1 vs robust_v2 verdicts on the user's NINJ1+FCHSD2 prompts.

Loads both profiles, runs each user-prompt file through both, prints a
side-by-side table. One-off scratch invocation (underscore-prefix).
"""
from __future__ import annotations

import sys
from pathlib import Path

from cairn.dotenv import load_dotenv
load_dotenv()

from cairn.pipeline import CairnPipeline  # noqa: E402

V1 = Path("configs/.profile_master_minus_arxiv_robust_v1.yaml")
V2 = Path("configs/.profile_master_minus_arxiv_robust_v2.yaml")

PROMPTS = [
    Path("data/_user_prompt_2026-05-17_A_judge_rebuttal.txt"),
    Path("data/_user_prompt_2026-05-17_B_solution.txt"),
    Path("data/_user_prompt_2026-05-17_C_error_analysis.txt"),
    Path("data/_user_prompt_2026-05-17_D_original_prompt.txt"),
    Path("data/_user_prompt_2026-05-17_E_combined_DCBA.txt"),
    Path("data/_user_prompt_2026-05-17b_A_solution.txt"),
    Path("data/_user_prompt_2026-05-17b_B_error_analysis.txt"),
    Path("data/_user_prompt_2026-05-17b_C_original_prompt.txt"),
    Path("data/_user_prompt_2026-05-17b_D_combined_CBA.txt"),
]

GROUND_TRUTH = {
    "A_judge_rebuttal": "LLM",
    "B_solution": "LLM",
    "C_error_analysis": "LLM",
    "D_original_prompt": "human",
    "E_combined_DCBA": "mixed",
    "b_A_solution": "LLM",
    "b_B_error_analysis": "LLM",
    "b_C_original_prompt": "?",
    "b_D_combined_CBA": "mixed",
}


def short_label(p: Path) -> str:
    return p.stem.replace("_user_prompt_2026-05-17", "").lstrip("_")


def main() -> None:
    print("Loading robust_v1...", flush=True)
    pipeline_v1 = CairnPipeline.from_config_file(V1)
    print("Loading robust_v2...", flush=True)
    pipeline_v2 = CairnPipeline.from_config_file(V2)

    print(f"\n{'Prompt':<28} {'truth':<7} {'v1.label':<10} {'v1.p_llm':<10} {'v2.label':<10} {'v2.p_llm':<10}")
    print("-" * 80)
    for path in PROMPTS:
        if not path.exists():
            print(f"{short_label(path):<28} MISSING")
            continue
        text = path.read_text(encoding="utf-8")
        det_v1 = pipeline_v1.analyze(text)
        det_v2 = pipeline_v2.analyze(text)
        label_short = short_label(path)
        truth = GROUND_TRUTH.get(label_short, "?")
        print(
            f"{label_short:<28} "
            f"{truth:<7} "
            f"{det_v1.label:<10} {det_v1.p_llm:<10.3f} "
            f"{det_v2.label:<10} {det_v2.p_llm:<10.3f}"
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
