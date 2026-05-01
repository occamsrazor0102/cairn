"""Render figures from the calib2000 benchmark CSVs.

Outputs PNGs to docs/benchmarks/calib2000/figures/. Designed to be
re-run any time the underlying CSVs change.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BENCH = Path("docs/benchmarks/calib2000")
OUT = BENCH / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TESTS = [
    "in_domain", "ood_m4", "ood_raid", "ood_mage", "verb_low",
    "ctrl_anthropic_t1", "ctrl_openai_med_high",
    "prompts_heldout", "prompts_fresh",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(s) -> float | None:
    if s in (None, "", "None"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def fig_profile_grid_heatmap() -> None:
    rows = read_csv(BENCH / "profile_grid_summary.csv")
    profiles = [r["profile"] for r in rows]
    matrix = np.zeros((len(profiles), len(TESTS)))
    for i, r in enumerate(rows):
        for j, t in enumerate(TESTS):
            v = fnum(r.get(f"{t}_auroc"))
            matrix[i, j] = v if v is not None else np.nan

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(TESTS)))
    ax.set_xticklabels(TESTS, rotation=45, ha="right")
    ax.set_yticks(range(len(profiles)))
    ax.set_yticklabels(profiles)
    ax.set_title("Profile grid - AUROC across 9 test sets\n"
                 "(13 profiles trained on overlapping subsets of the calib2000 v6 master corpus)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.65 or v > 0.92 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(OUT / "profile_grid_auroc.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_loo_deltas() -> None:
    rows = read_csv(BENCH / "loo_summary.csv")
    grid = {r["profile"]: r for r in read_csv(BENCH / "profile_grid_summary.csv")}
    everything = grid.get("everything")
    if not everything:
        return

    sources, deltas = [], []
    for r in rows:
        diffs = []
        for t in TESTS:
            v = fnum(r.get(f"{t}_auroc"))
            base = fnum(everything.get(f"{t}_auroc"))
            if v is not None and base is not None:
                diffs.append(v - base)
        if diffs:
            sources.append(r["excluded_source"])
            deltas.append(float(np.mean(diffs)))

    order = np.argsort(deltas)[::-1]
    sources = [sources[i] for i in order]
    deltas = [deltas[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
    ax.barh(range(len(sources)), deltas, color=colors)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean AUROC delta vs `everything` baseline (across 9 test sets)")
    ax.set_title("Per-source leave-one-out - sources whose REMOVAL helps (green) or hurts (red)\n"
                 "Top entry: drop arxiv -> +0.026 mean improvement")
    for i, d in enumerate(deltas):
        ax.text(d + (0.001 if d > 0 else -0.001), i,
                f"{d:+.4f}",
                va="center", ha="left" if d > 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "loo_deltas.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _load_profile_aurocs(profile: str, grid: dict) -> list[float]:
    """Get AUROC list across TESTS for a profile, falling back to JSON benchmark dir."""
    if profile in grid:
        return [fnum(grid[profile].get(f"{t}_auroc")) or 0.0 for t in TESTS]
    out = []
    for t in TESTS:
        p = BENCH / "profiles" / profile / f"{t}.json"
        if p.exists():
            import json
            r = json.loads(p.read_text())
            out.append(float(r.get("metrics", {}).get("auroc") or 0.0))
        else:
            out.append(0.0)
    return out


def fig_master_vs_baselines() -> None:
    grid = {r["profile"]: r for r in read_csv(BENCH / "profile_grid_summary.csv")}
    profs = ["general_hc3", "everything", "master_minus_arxiv", "drop_top3_drags"]
    aurocs = {p: _load_profile_aurocs(p, grid) for p in profs}

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(TESTS))
    width = 0.2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for i, p in enumerate(profs):
        ax.bar(x + (i - 1.5) * width, aurocs[p], width, label=p, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(TESTS, rotation=30, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Production profile comparison - `master_minus_arxiv` recommended\n"
                 "Beats `everything` on 6/9 with ~zero in-domain cost (+0.0004)")
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "production_profile_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_per_detector_auroc() -> None:
    path = BENCH / "per_detector_auroc.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    if not rows:
        return
    datasets = [k for k in rows[0].keys() if k != "detector"]
    detectors = [r["detector"] for r in rows]

    matrix = np.full((len(detectors), len(datasets)), np.nan)
    for i, r in enumerate(rows):
        for j, d in enumerate(datasets):
            v = fnum(r.get(d))
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(detectors))))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_yticks(range(len(detectors)))
    ax.set_yticklabels(detectors)
    ax.set_title("Per-detector AUROC across datasets\n"
                 "(0.5 = random; <0.5 = inverted signal)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.55 or v > 0.92 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(OUT / "per_detector_auroc.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_verb_low_attack() -> None:
    grid = {r["profile"]: r for r in read_csv(BENCH / "profile_grid_summary.csv")}
    profs_show = ["general_hc3", "general_full", "everything",
                  "prompts_only", "master_minus_arxiv", "drop_top3_drags"]
    in_dom = []
    verb = []
    for p in profs_show:
        a = _load_profile_aurocs(p, grid)
        in_dom.append(a[TESTS.index("in_domain")])
        verb.append(a[TESTS.index("verb_low")])

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(profs_show))
    ax.bar(x - 0.2, in_dom, 0.4, label="in_domain (HC3 prose, default verbosity)", color="#1f77b4")
    ax.bar(x + 0.2, verb, 0.4, label="verb_low (OpenAI Responses, verbosity=low)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(profs_show, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("verbosity=low is a primary attack surface\n"
                 "Length-aware EBM features partially recover it on master_minus_arxiv / drop_top3_drags")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "verb_low_attack.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_loss_filter_negative() -> None:
    path = BENCH / "loss_filter_summary.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    if not rows:
        return
    grid = {r["profile"]: r for r in read_csv(BENCH / "profile_grid_summary.csv")}
    base_auroc = fnum(grid.get("everything", {}).get("in_domain_auroc")) or 0.0

    strategies = [r["strategy"] for r in rows]
    in_dom = [fnum(r.get("in_domain_auroc")) or 0 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(["everything\n(baseline)", *strategies], [base_auroc, *in_dom],
                  color=["#1f77b4", "#d62728", "#d62728", "#ff7f0e", "#ff7f0e"])
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="random (AUROC=0.5)")
    ax.set_ylabel("in_domain AUROC")
    ax.set_title("Loss-filter point selection - NEGATIVE result\n"
                 "All four strategies underperform the full corpus; high_loss inverts the detector")
    ax.set_ylim(0, 1.0)
    for bar, v in zip(bars, [base_auroc, *in_dom]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=9)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "loss_filter_negative.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_profile_grid_heatmap()
    fig_loo_deltas()
    fig_master_vs_baselines()
    fig_per_detector_auroc()
    fig_verb_low_attack()
    fig_loss_filter_negative()
    print(f"Wrote figures to {OUT}/")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
