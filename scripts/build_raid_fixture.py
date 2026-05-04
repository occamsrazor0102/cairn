"""One-shot builder: pull a deterministic 50-row sample from the public RAID test CSV."""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", default=Path("tests/fixtures/raid_mini.jsonl"), type=Path)
    ap.add_argument("--n-per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    with args.source.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row.get("label", 0))
            text = row.get("generation") or row.get("text") or ""
            if not text or len(text.split()) < 50:
                continue
            by_label[label].append({
                "id": f"raid_{row.get('id') or row.get('row_id')}",
                "text": text, "label": label,
                "source": row.get("model", "raid_human") or "raid_human",
                "attack_name": row.get("decoding") or row.get("attack"),
                "tier": "raid",
            })
    out_rows = (
        rng.sample(by_label[0], args.n_per_class) +
        rng.sample(by_label[1], args.n_per_class)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
