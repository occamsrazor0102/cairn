"""Project Lighthouse calibration probe.

Run every item (A1..A20, B1..B20) from the two Lighthouse detection-screen
.docx files through CAIRN's calibrated `prompts_only` profile. All 40 items
are KNOWN LLM-generated, so this is a false-negative / calibration probe:
how many does CAIRN flag as LLM, and at what p(LLM)?

Nothing leaves the machine: prompts_only extends the default cascade with the
trained EBM + conformal + detector-calibration artifacts; Phase-3 API
detectors stay off and task_description auto-mode is off.

Usage:  python scripts/_lighthouse_screen_cairn.py
"""
import json
import re
import sys
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from cairn.config import load_config  # noqa: E402
from cairn.pipeline import CairnPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = {
    "A": r"C:\Users\occam\Documents\Codex\2026-06-26\new-chat\outputs\project_lighthouse_screen_form_A.docx",
    "B": r"C:\Users\occam\Documents\Codex\2026-06-26\new-chat\outputs\project_lighthouse_screen_form_B.docx",
}
LABEL_RE = re.compile(r"^[AB]\d{1,2}$")


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _paragraphs(docx_path: str) -> list[str]:
    z = zipfile.ZipFile(docx_path)
    xml = _decode(z.read("word/document.xml"))
    out = []
    for para in re.split(r"</w:p>", xml):
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S)
        line = re.sub(r"<[^>]+>", "", "".join(texts))
        # unescape the handful of XML entities Word emits
        line = (line.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'"))
        if line.strip():
            out.append(line.strip())
    return out


def extract_items() -> dict[str, str]:
    """Return {label: prompt_text} for every A#/B# item across both forms."""
    items: dict[str, str] = {}
    for form, path in DOCS.items():
        paras = _paragraphs(path)
        cur = None
        buf: list[str] = []
        for p in paras:
            if LABEL_RE.match(p):
                if cur is not None:
                    items[cur] = " ".join(buf).strip()
                cur = p
                buf = []
            elif cur is not None:
                buf.append(p)
        if cur is not None:
            items[cur] = " ".join(buf).strip()
    return items


def _sort_key(label: str) -> tuple[str, int]:
    return (label[0], int(label[1:]))


def _j(obj):
    if is_dataclass(obj):
        return _j(asdict(obj))
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    return obj


def main() -> int:
    items = extract_items()
    labels = sorted(items, key=_sort_key)
    print(f"extracted {len(labels)} items: {labels}")
    assert len(labels) == 40, f"expected 40 items, got {len(labels)}"

    cfg = load_config("configs/.profile_prompts_only.yaml")
    pipe = CairnPipeline(cfg)

    rows = []
    for lab in labels:
        text = items[lab]
        det, results, router = pipe.analyze_detailed(text)
        d = _j(det)
        # top contributing detectors by p_llm (only ones that actually fired)
        fired = [
            (r_.get("layer_id"), r_.get("p_llm"))
            for r_ in (_j(x) for x in results)
            if isinstance(r_.get("p_llm"), (int, float))
            and not (r_.get("signals") or {}).get("skipped")
        ]
        fired.sort(key=lambda t: t[1], reverse=True)
        rows.append({
            "label": lab,
            "words": len(text.split()),
            "p_llm": d.get("p_llm"),
            "decision": d.get("label"),
            "conformal_set": d.get("conformal_set"),
            "prediction_set": d.get("prediction_set"),
            "calibration_status": d.get("calibration_status"),
            "router_domain": router.domain,
            "cascade_phases": d.get("cascade_phases"),
            "top_detectors": fired[:4],
            "detector_errors": d.get("detector_errors"),
            "text": text,
        })

    # ---------- report ----------
    print("\ncalibration_status: " + str(rows[0]["calibration_status"]))
    print("=" * 92)
    print(f"{'item':5s} {'words':>5s} {'p_llm':>7s} {'decision':>9s} "
          f"{'conformal_set':>20s}  top detector")
    print("-" * 92)
    for r in rows:
        pl = r["p_llm"]
        pls = f"{pl:.4f}" if isinstance(pl, (int, float)) else str(pl)
        top = r["top_detectors"][0] if r["top_detectors"] else ("-", 0)
        top_s = f"{top[0]}={top[1]:.2f}" if isinstance(top[1], (int, float)) else "-"
        cs = ",".join(r["conformal_set"]) if r["conformal_set"] else "-"
        print(f"{r['label']:5s} {r['words']:5d} {pls:>7s} {str(r['decision']):>9s} "
              f"{cs:>20s}  {top_s}")

    # ---------- summary (all are KNOWN LLM) ----------
    pvals = [r["p_llm"] for r in rows if isinstance(r["p_llm"], (int, float))]
    pvals_sorted = sorted(pvals)
    n = len(pvals)

    def pct(p):
        if not pvals_sorted:
            return float("nan")
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return pvals_sorted[k]

    called_llm = sum(1 for r in rows if str(r["decision"]).upper() == "LLM")
    called_human = sum(1 for r in rows if str(r["decision"]).lower() == "human")
    # conformal: does the honest set CONTAIN LLM? (covers singleton + abstain)
    conf_has_llm = sum(1 for r in rows if r["conformal_set"] and "LLM" in r["conformal_set"])
    conf_singleton_llm = sum(1 for r in rows if r["conformal_set"] == ["LLM"])
    conf_human_only = sum(1 for r in rows if r["conformal_set"] == ["human"])

    print("\n" + "=" * 92)
    print("SUMMARY  (ground truth: all 40 items are LLM-generated)")
    print("=" * 92)
    print(f"  decision == LLM     : {called_llm}/40   ({called_llm/40:.0%})  <- correct")
    print(f"  decision == human   : {called_human}/40   ({called_human/40:.0%})  <- FALSE NEGATIVES")
    print(f"  conformal set ∋ LLM : {conf_has_llm}/40")
    print(f"    of which singleton ['LLM']     : {conf_singleton_llm}/40")
    print(f"  conformal set == ['human'] only  : {conf_human_only}/40  <- confident misses")
    if pvals:
        print(f"  p_llm  mean={sum(pvals)/n:.3f}  min={min(pvals):.3f}  "
              f"p25={pct(.25):.3f}  median={pct(.5):.3f}  p75={pct(.75):.3f}  max={max(pvals):.3f}")
    # per-form split
    for form in ("A", "B"):
        fr = [r for r in rows if r["label"].startswith(form)]
        fp = [r["p_llm"] for r in fr if isinstance(r["p_llm"], (int, float))]
        cl = sum(1 for r in fr if str(r["decision"]).upper() == "LLM")
        if fp:
            print(f"  form {form}: decision==LLM {cl}/{len(fr)}  mean p_llm={sum(fp)/len(fp):.3f}")

    errs = [(r["label"], r["detector_errors"]) for r in rows if r["detector_errors"]]
    if errs:
        print("\n  detector_errors observed:")
        for lab, e in errs[:8]:
            print(f"    {lab}: {e}")

    out = ROOT / "scripts" / "_lighthouse_screen_results.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
