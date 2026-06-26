"""Project Lighthouse 31-variant graded benchmark -> CAIRN calibration.

Runs all 62 benchmark items through the calibrated prompts_only profile and
compares CAIRN's p(LLM) to the human blind-panel tiers/scores:

  - 31 raw variants            (project_lighthouse_prompt_variants_expanded.md)
  - 10 Tier 1 "genuine"        (humanized_prompt_rewrites_top10.md)        blind ~75
  - 11 Tier 2 "performed"      (detection_tiers_...md  SET A)              blind ~50
  - 10 Tier 3 "blatant LLM"    (detection_tiers_...md  SET B)              blind ~7

Ground truth: ALL are AI-generated/AI-transformed (none genuine-human); the
blind score is how-HUMAN each reads (0=def-AI, 100=def-human). So we expect
CAIRN p(LLM) to fall as the blind score rises, and to order tiers T3>raw>T2>T1.

prompts_only loads via skops (no legacy flag). Nothing leaves the machine.
Run:  python scripts/_lighthouse_benchmark_cairn.py
"""
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from cairn.config import load_config  # noqa: E402
from cairn.pipeline import CairnPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(r"C:\Users\occam\Documents\Codex\2026-06-26\new-chat\outputs")
F_T1 = SRC / "humanized_prompt_rewrites_top10.md"
F_T23 = SRC / "detection_tiers_performed-human_and_blatant-LLM.md"
F_RAW = SRC / "project_lighthouse_prompt_variants_expanded.md"
F_KEY = SRC / "detection_benchmark_31variants.md"

VID = r"(?:Expanded|Alt|Add)\s*\d+"


def _read(p: Path) -> str:
    raw = p.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _norm_id(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _raw_scores() -> dict[str, int]:
    """Parse the answer-key table for each variant's RAW blind score."""
    out = {}
    for line in _read(F_KEY).splitlines():
        m = re.match(r"\|\s*(" + VID + r")\s*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|", line)
        if m:
            out[_norm_id(m.group(1))] = int(m.group(2))
    return out


def extract_tier1() -> list[dict]:
    items = []
    lines = _read(F_T1).splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^###\s*\d+\.\s*(.+?)\s*[—-]\s*(" + VID + r")\s*\((\d+)\s*[→-]+\s*(\d+)\)", lines[i])
        if m:
            persona, vid, score = m.group(1).strip(), _norm_id(m.group(2)), int(m.group(4))
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("### "):
                if lines[i].startswith(">"):
                    body.append(lines[i].lstrip("> ").rstrip())
                i += 1
            items.append({"variant_id": vid, "persona": persona, "tier": "T1_genuine",
                          "blind_score": score, "text": " ".join(x for x in body if x).strip()})
        else:
            i += 1
    return items


def extract_tier23() -> list[dict]:
    items = []
    lines = _read(F_T23).splitlines()
    cur_set = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if re.match(r"^#\s*SET\s*A", ln):
            cur_set = "T2_performed"
        elif re.match(r"^#\s*SET\s*B", ln):
            cur_set = "T3_blatant"
        m = re.match(r"^###\s*([AB]\d+)\s*[·.]\s*(" + VID + r")\s*[—-]\s*(.+?)\s*[—-]\s*score\s*(\d+)", ln)
        if m and cur_set:
            vid, persona, score = _norm_id(m.group(2)), m.group(3).strip(), int(m.group(4))
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("###") and not lines[i].startswith("# SET") and not lines[i].startswith("---"):
                if lines[i].strip():
                    body.append(lines[i].strip())
                i += 1
            items.append({"variant_id": vid, "persona": persona, "tier": cur_set,
                          "blind_score": score, "text": " ".join(body).strip()})
        else:
            i += 1
    return items


def extract_raw() -> list[dict]:
    items = []
    raw_scores = _raw_scores()
    lines = _read(F_RAW).splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^###\s*(" + VID + r")\s*-\s*(.+)", lines[i])
        if m:
            vid, persona = _norm_id(m.group(1)), m.group(2).strip()
            body = []
            i += 1
            while i < len(lines) and not re.match(r"^#{2,3}\s", lines[i]):
                s = lines[i].strip()
                if s and not s.startswith("Target category:") and s != "Prompt:":
                    body.append(s)
                i += 1
            items.append({"variant_id": vid, "persona": persona, "tier": "raw",
                          "blind_score": raw_scores.get(vid), "text": " ".join(body).strip()})
        else:
            i += 1
    return items


def main() -> int:
    raw = extract_raw()
    t1 = extract_tier1()
    t23 = extract_tier23()
    t2 = [x for x in t23 if x["tier"] == "T2_performed"]
    t3 = [x for x in t23 if x["tier"] == "T3_blatant"]
    print(f"extracted: raw={len(raw)}  T1={len(t1)}  T2={len(t2)}  T3={len(t3)}  total={len(raw)+len(t1)+len(t2)+len(t3)}")
    assert len(raw) == 31, f"raw expected 31, got {len(raw)}"
    assert len(t1) == 10 and len(t2) == 11 and len(t3) == 10, "tier counts off"

    allitems = []
    for grp, lst in (("raw", raw), ("T1", t1), ("T2", t2), ("T3", t3)):
        for k, it in enumerate(lst, 1):
            it = dict(it)
            it["id"] = f"lh_{grp}_{k:02d}_{it['variant_id'].replace(' ', '')}"
            it["words"] = len(it["text"].split())
            allitems.append(it)
    # sanity: show shortest extraction (catch a botched parse)
    shortest = min(allitems, key=lambda x: x["words"])
    print(f"shortest item: {shortest['id']} ({shortest['words']} words): {shortest['text'][:70]!r}")

    cfg = load_config("configs/.profile_prompts_only.yaml")
    pipe = CairnPipeline(cfg)
    print("calibration_status:", pipe._calibration_status)

    for it in allitems:
        det = pipe.analyze(it["text"])
        it["p_llm"] = det.p_llm
        it["band"] = det.label
        it["conformal_set"] = det.conformal_set
        it["top_detector"] = None
        # (we don't need per-detector here; keep payload small)

    out = ROOT / "scripts" / "_lighthouse_benchmark_results.json"
    out.write_text(json.dumps(allitems, indent=2, default=str), encoding="utf-8")

    # labeled calibration slice
    slice_path = ROOT / "data" / "lighthouse_benchmark_eval.jsonl"
    with open(slice_path, "w", encoding="utf-8") as f:
        for it in allitems:
            f.write(json.dumps({
                "id": it["id"], "text": it["text"], "label": 1, "tier": it["tier"],
                "blind_human_score": it["blind_score"], "variant_id": it["variant_id"],
                "persona": it["persona"], "source": "lighthouse_benchmark",
            }, ensure_ascii=False) + "\n")

    # ---------- analysis ----------
    def stats(lst, name, blind_target):
        p = [x["p_llm"] for x in lst]
        bs = [x["blind_score"] for x in lst if x["blind_score"] is not None]
        ge = sum(1 for v in p if v >= 0.5)
        cL = sum(1 for x in lst if x["conformal_set"] == ["LLM"])
        cH = sum(1 for x in lst if x["conformal_set"] == ["human"])
        bs_m = f"{sum(bs)/len(bs):5.1f}" if bs else "  n/a"
        print(f"  {name:12s} n={len(lst):2d}  blind={bs_m}{blind_target:>9s}  "
              f"p_llm mean={sum(p)/len(p):.3f} median={sorted(p)[len(p)//2]:.3f}  "
              f"flag(p>=.5)={ge:2d}/{len(lst)}  conf[LLM]={cL:2d} conf[human]={cH:2d}")

    print("\n" + "=" * 100)
    print("CAIRN vs blind human panel  (ground truth: ALL items are AI; blind score = how HUMAN it reads)")
    print("=" * 100)
    stats(allitems, "ALL", "")
    stats([x for x in allitems if x["tier"] == "T3_blatant"], "T3 blatant", "  (~7)")
    stats([x for x in allitems if x["tier"] == "raw"], "raw (31)", " (11-57)")
    stats([x for x in allitems if x["tier"] == "T2_performed"], "T2 performed", " (~50)")
    stats([x for x in allitems if x["tier"] == "T1_genuine"], "T1 genuine", "  (~75)")

    # correlation p_llm vs blind humanness (expect strong NEGATIVE)
    paired = [(x["p_llm"], x["blind_score"]) for x in allitems if x["blind_score"] is not None]
    n = len(paired)

    def pearson(xs, ys):
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        vx = sum((a - mx) ** 2 for a in xs) ** 0.5
        vy = sum((b - my) ** 2 for b in ys) ** 0.5
        return cov / (vx * vy) if vx and vy else float("nan")

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for pos, idx in enumerate(order):
            r[idx] = pos
        return r

    px = [a for a, _ in paired]
    py = [b for _, b in paired]
    pear = pearson(px, py)
    spear = pearson(rank(px), rank(py))
    print("-" * 100)
    print(f"  p_llm vs blind humanness (n={n}):  Pearson r={pear:+.3f}   Spearman rho={spear:+.3f}  "
          "(negative = CAIRN agrees: more-human-reading -> lower p_llm)")
    print(f"\nwrote {out}")
    print(f"wrote {slice_path}  ({len(allitems)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
