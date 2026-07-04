"""Print first record from v7 and v8 caches to discover the actual schema."""
import json
from pathlib import Path

for p in [Path("data/all_features_v7.jsonl"), Path("data/all_features_v8.jsonl")]:
    print(f"=== {p} ===")
    with p.open("r", encoding="utf-8") as fh:
        rec = json.loads(fh.readline())
    print(f"keys ({len(rec)}):")
    for k in sorted(rec.keys()):
        v = rec[k]
        t = type(v).__name__
        s = repr(v)
        if len(s) > 80:
            s = s[:80] + "..."
        print(f"  {k!r:<40} {t:<10} {s}")
    print()
