#!/usr/bin/env python3
"""
test_interlinear.py — do a book's tsawa spans, concatenated in order,
reconstruct a continuous root text?

The claim: some commentaries are mchan-'grel (interlinear), where the author
weaves prose between the individual words of the root verse rather than
quoting a line and then explaining it. If so, the very short spans in those
books are correct annotation of a different commentary form, and stitching
them back together should yield readable root text.

If that holds it explains a lot: a token classifier asked to place boundaries
around thousands of isolated one-syllable spans has almost nothing local to go
on, which matches the 0.037 argmax span F1.

The check is qualitative and has to be read, not scored. Two signals help:

  METRE — if the stitched text is verse, its shad-delimited clauses should be
  isometric (7/9/11/13 syllables), like the rest of the corpus (88.3%).

  GAPS — interlinear commentary means small gaps between consecutive spans.
  Block quotation means long gaps with whole paragraphs in between.

Usage
-----
    python test_interlinear.py --pecha IF3ACC3E1
    python test_interlinear.py --compare          # an interlinear-looking book
                                                  # against a normal one
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

TSHEG = "\u0f0b"
SHADS = "\u0f0d\u0f0e\u0f0f\u0f10\u0f11\u0f14"

SUSPECT = ["IF3ACC3E1", "I881A57E8", "I9FF2B59B", "I1637B774",
           "IDAD44BA2", "IB522F095"]


def syllables(s: str) -> int:
    return len([p for p in s.split(TSHEG) if p.strip()])


def load(raw: Path, pid: str) -> str | None:
    for c in (raw / f"{pid}.opf" / f"{pid}.opf" / "base" / "v001.txt",
              raw / f"{pid}.opf" / "base" / "v001.txt"):
        if c.is_file():
            return c.read_text(encoding="utf-8")
    return None


def analyse(text: str, spans: pd.DataFrame, pid: str, n_show: int):
    spans = spans.sort_values("start")
    lens = (spans.end - spans.start).tolist()
    gaps = []
    prev = None
    for _, r in spans.iterrows():
        if prev is not None:
            gaps.append(int(r.start) - prev)
        prev = int(r.end)

    print(f"\n{'='*66}")
    print(f"{pid} — {len(spans):,} spans")
    print("=" * 66)
    print(f"  span length: median {statistics.median(lens):.0f} chars, "
          f"mean {statistics.mean(lens):.0f}")
    if gaps:
        print(f"  gap between spans: median {statistics.median(gaps):.0f} chars")
        tiny = sum(1 for g in gaps if 0 <= g <= 30)
        print(f"  gaps <= 30 chars: {tiny:,} ({100*tiny/len(gaps):.1f}%)")
        print("    high share -> interlinear; low -> block quotation")

    # ---- stitch ----------------------------------------------------------
    stitched = "".join(text[int(r.start):int(r.end)] for _, r in spans.iterrows())
    print(f"\n  stitched length: {len(stitched):,} chars "
          f"({100*len(stitched)/len(text):.1f}% of the book)")

    clauses = [c for c in
               "".join(ch if ch not in SHADS else "\n" for ch in stitched).split("\n")
               if c.strip()]
    counts = [syllables(c) for c in clauses if 1 <= syllables(c) <= 30]
    if counts:
        iso = sum(1 for n in counts if n in (7, 9, 11, 13))
        print(f"  stitched clauses: {len(counts):,}, median {statistics.median(counts)}, "
              f"stdev {statistics.pstdev(counts):.1f}")
        print(f"  isometric (7/9/11/13): {100*iso/len(counts):.1f}%")
        print(f"    corpus tsawa average is 88.3%; commentary is 36.7%.")
        print(f"    If this is genuinely root verse stitched back together,")
        print(f"    it should look like the former.")
        print(f"  top clause lengths: " +
              ", ".join(f"{k}({v})" for k, v in Counter(counts).most_common(6)))

    print(f"\n  --- first {n_show} chars of the stitched text ---")
    print("  " + stitched[:n_show].replace("\n", " "))
    print(f"\n  --- the same region in the book, spans marked ---")
    first = spans.head(12)
    lo = int(first.iloc[0].start)
    hi = int(first.iloc[-1].end)
    hi = min(hi, lo + 600)
    out, cur = [], lo
    for _, r in first.iterrows():
        s, e = int(r.start), int(r.end)
        if s >= hi:
            break
        out.append(text[cur:s])
        out.append("«" + text[s:e] + "»")
        cur = e
    out.append(text[cur:hi])
    print("  " + "".join(out).replace("\n", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="tsawa/data/processed/tsawa_spans_resolved.csv")
    ap.add_argument("--raw-opf", default="data/raw_opf")
    ap.add_argument("--pecha", default=None)
    ap.add_argument("--compare", action="store_true",
                    help="one suspected interlinear book vs one normal book")
    ap.add_argument("--chars", type=int, default=400)
    args = ap.parse_args()

    raw = Path(args.raw_opf)
    d = pd.read_csv(args.spans)
    if "dropped" in d.columns:
        d = d[~d.dropped.astype(bool)]
    d = d.copy()
    d["len"] = d.end - d.start

    if args.pecha:
        pids = [args.pecha]
    elif args.compare:
        # a suspect book, plus a book whose spans are long and normal
        normal = (d.groupby("pecha_id")["len"].median()
                  .sort_values(ascending=False))
        normal = [p for p in normal.index if p not in SUSPECT][:1]
        pids = ["IF3ACC3E1"] + normal
    else:
        pids = SUSPECT[:2]

    for pid in pids:
        t = load(raw, pid)
        if t is None:
            print(f"  (no base text for {pid})")
            continue
        sp = d[d.pecha_id == pid]
        if not len(sp):
            print(f"  (no spans for {pid})")
            continue
        analyse(t, sp, pid, args.chars)

    print(f"\n{'='*66}")
    print("HOW TO READ THIS")
    print("=" * 66)
    print("  If the stitched text is coherent root verse — isometric clauses,")
    print("  reading continuously — then these books are interlinear")
    print("  commentary and the short spans are correct. They need monotonic")
    print("  alignment, not better token classification.")
    print()
    print("  If the stitched text is disjointed fragments that do not read as")
    print("  anything, the spans are noise and the interlinear theory is wrong.")
    print()
    print("  This needs a reader. The metre percentage is a hint, not a verdict.")


if __name__ == "__main__":
    main()