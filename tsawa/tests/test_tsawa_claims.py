#!/usr/bin/env python3
"""
test_tsawa_claims.py — check four structural claims against the annotations.

The claims (from a Tibetanist-style description of commentary structure) are
plausible but unverified. Each one, if true, is a constraint or a feature the
model does not currently have. Each is cheap to check.

  1. SEQUENTIALITY. Tsawa fragments appear in the order of the original root
     text, so consecutive tsawa spans in a commentary should read as
     continuous text. If true, a decoder could enforce monotonic alignment —
     a constraint nothing in the pipeline uses today.

  2. QUOTE PARTICLES SIT OUTSIDE THE SPAN. Spans should end just before
     ཞེས་/ཅེས་, not include them. If the corpus is inconsistent here, boundary
     F1 has a ceiling no model can pass.

  3. METRE. Verse root text has regular syllable counts between shad — 7, 9,
     11, 13. Commentary prose does not. If tsawa clauses cluster on those
     counts, clause length is a strong feature.

  4. WORD-BY-WORD COMMENTARY IS A REAL GENRE. In tshig-'grel, the root stanza
     is broken into 2-4 syllable lemmas, each followed by explanation. If the
     books with many very short spans show that alternating pattern, those
     spans are correct annotation of a different commentary type — not the
     noise they were assumed to be.

Nothing is modified. Usage:

    python test_tsawa_claims.py
    python test_tsawa_claims.py --books 40 --examples 6
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

TSHEG = "\u0f0b"
SHADS = "\u0f0d\u0f0e\u0f0f\u0f10\u0f11\u0f14"
CLOSERS = ["ཞེས་པ་སྟེ", "ཞེས་པ་ནི", "ཞེས་པའོ", "ཞེས་གསུངས", "ཅེས་གསུངས",
           "ཞེས་བྱ་བ", "ཞེས་པ", "ཅེས་པ", "ཞེས་", "ཅེས་", "གསུངས་"]
# word-by-word commentary marker: lemma followed by "means..."
LEMMA_MARK = ["ཞེས་པ་ནི", "ཅེས་པ་ནི", "ཞེས་བྱ་བ་ནི"]


def syllables(s: str) -> int:
    """Rough syllable count: tsheg-delimited units."""
    s = s.strip()
    if not s:
        return 0
    return len([p for p in s.split(TSHEG) if p.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="tsawa/data/processed/tsawa_spans_resolved.csv")
    ap.add_argument("--raw-opf", default="data/raw_opf")
    ap.add_argument("--books", type=int, default=40)
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    raw = Path(args.raw_opf)
    d = pd.read_csv(args.spans)
    if "dropped" in d.columns:
        d = d[~d.dropped.astype(bool)]
    d = d.copy()
    d["len"] = d.end - d.start

    pids = sorted(d.pecha_id.unique())
    import random
    random.seed(args.seed)
    random.shuffle(pids)

    texts = {}
    for pid in pids:
        for c in (raw / f"{pid}.opf" / f"{pid}.opf" / "base" / "v001.txt",
                  raw / f"{pid}.opf" / "base" / "v001.txt"):
            if c.is_file():
                texts[pid] = c.read_text(encoding="utf-8")
                break
        if len(texts) >= args.books:
            break
    print(f"checking {len(texts)} books\n")

    # =====================================================================
    print("=" * 66)
    print("1. SEQUENTIALITY — do consecutive tsawa spans read as continuous text?")
    print("=" * 66)
    print("  If tsawa is the root text quoted in order, then span N's text")
    print("  should continue where span N-1 left off, ignoring commentary")
    print("  in between. Hard to check exactly, so this uses a proxy: are")
    print("  spans non-overlapping and strictly increasing in offset, and")
    print("  how much commentary sits between them?\n")
    gaps_all = []
    for pid, t in texts.items():
        sp = d[d.pecha_id == pid].sort_values("start")
        prev_end = None
        for _, r in sp.iterrows():
            if prev_end is not None:
                gaps_all.append(int(r.start) - prev_end)
            prev_end = int(r.end)
    if gaps_all:
        neg = sum(1 for g in gaps_all if g < 0)
        print(f"  span pairs examined: {len(gaps_all):,}")
        print(f"  out-of-order / overlapping: {neg} ({100*neg/len(gaps_all):.2f}%)")
        print(f"  gap between consecutive spans — median {statistics.median(gaps_all):.0f} "
              f"chars, mean {statistics.mean(gaps_all):.0f}")
        tiny = sum(1 for g in gaps_all if 0 <= g <= 30)
        print(f"  gaps of 30 chars or less: {tiny:,} ({100*tiny/len(gaps_all):.1f}%)")
        print("    -> a large share of tiny gaps means word-by-word commentary,")
        print("       where lemma and explanation alternate closely.")

    # =====================================================================
    print("\n" + "=" * 66)
    print("2. QUOTE PARTICLES — inside the span or outside it?")
    print("=" * 66)
    ends_with = Counter()
    followed_by = Counter()
    checked = 0
    for pid, t in texts.items():
        for _, r in d[d.pecha_id == pid].iterrows():
            s, e = int(r.start), int(r.end)
            span = t[s:e]
            after = t[e:e + 14]
            checked += 1
            hit_in = next((c for c in CLOSERS if span.rstrip().endswith(c)), None)
            if hit_in:
                ends_with[hit_in] += 1
            hit_after = next((c for c in CLOSERS
                              if after.lstrip(" \n།").startswith(c)), None)
            if hit_after:
                followed_by[hit_after] += 1
    n_in = sum(ends_with.values())
    n_after = sum(followed_by.values())
    print(f"  spans checked: {checked:,}")
    print(f"  span ENDS WITH a quote particle:    {n_in:,} "
          f"({100*n_in/max(checked,1):.1f}%)")
    print(f"  particle appears just AFTER the span: {n_after:,} "
          f"({100*n_after/max(checked,1):.1f}%)")
    if n_after > n_in * 3:
        print("  -> convention is consistent: particle OUTSIDE the span.")
    elif n_in > n_after * 3:
        print("  -> convention is consistent: particle INSIDE the span.")
    elif n_in + n_after > 0:
        print("  -> INCONSISTENT. Both conventions appear. This alone caps")
        print("     boundary F1: the model cannot satisfy both.")
    print(f"  most common particles after a span: "
          f"{', '.join(k for k, _ in followed_by.most_common(4))}")

    # =====================================================================
    print("\n" + "=" * 66)
    print("3. METRE — are tsawa clauses isometric (7/9/11/13 syllables)?")
    print("=" * 66)
    tsawa_clauses, other_clauses = [], []
    for pid, t in texts.items():
        mask = bytearray(len(t))
        for _, r in d[d.pecha_id == pid].iterrows():
            for i in range(int(r.start), min(int(r.end), len(t))):
                mask[i] = 1
        for m in re.finditer(rf"[^{SHADS}]+", t):
            seg = m.group()
            n = syllables(seg)
            if not (1 <= n <= 30):
                continue
            frac = sum(mask[m.start():m.end()]) / max(m.end() - m.start(), 1)
            (tsawa_clauses if frac > 0.8 else other_clauses).append(n)

    def profile(xs, name):
        if not xs:
            return
        c = Counter(xs)
        iso = sum(c[k] for k in (7, 9, 11, 13))
        print(f"  {name}: {len(xs):,} clauses, median {statistics.median(xs)}, "
              f"stdev {statistics.pstdev(xs):.1f}")
        print(f"    on 7/9/11/13 syllables: {100*iso/len(xs):.1f}%")
        print(f"    top lengths: " +
              ", ".join(f"{k}({v})" for k, v in c.most_common(6)))
    profile(tsawa_clauses, "inside tsawa ")
    profile(other_clauses, "outside tsawa")
    if tsawa_clauses and other_clauses:
        ti = sum(Counter(tsawa_clauses)[k] for k in (7, 9, 11, 13)) / len(tsawa_clauses)
        oi = sum(Counter(other_clauses)[k] for k in (7, 9, 11, 13)) / len(other_clauses)
        print(f"\n  isometric lift inside vs outside: {ti/max(oi,1e-9):.2f}x")
        print("    -> above ~1.5x means clause syllable count is a real feature.")

    # =====================================================================
    print("\n" + "=" * 66)
    print("4. WORD-BY-WORD COMMENTARY — is the short-span pattern a genre?")
    print("=" * 66)
    print("  Claim: in tshig-'grel the root stanza is split into 2-4 syllable")
    print("  lemmas, each followed by 'X ཞེས་པ་ནི' plus explanation. If so, the")
    print("  books full of very short spans are correctly annotated, not noisy.\n")
    for pid in ["I881A57E8", "I9FF2B59B", "IF3ACC3E1"]:
        if pid not in texts:
            continue
        t = texts[pid]
        sp = d[(d.pecha_id == pid) & (d["len"] <= 20)].sort_values("start")
        if not len(sp):
            continue
        lemma_follow = 0
        for _, r in sp.iterrows():
            after = t[int(r.end):int(r.end) + 16].lstrip(" \n།")
            if any(after.startswith(m) for m in LEMMA_MARK):
                lemma_follow += 1
        print(f"  {pid}: {len(sp):,} short spans, "
              f"{lemma_follow:,} followed by a lemma marker "
              f"({100*lemma_follow/len(sp):.1f}%)")
        for _, r in sp.head(args.examples).iterrows():
            s, e = int(r.start), int(r.end)
            print(f"      «{t[s:e]}» {t[e:e+45]}".replace("\n", " "))
        print()
    print("  A high percentage would mean these spans are a recognised")
    print("  commentary form and should be modelled, not excluded.")


if __name__ == "__main__":
    main()
    