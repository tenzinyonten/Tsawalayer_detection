#!/usr/bin/env python3
"""
test_repetition_signal.py — is "appears in another book" a usable tsawa detector?

Commentary prose is written once by one author. Root verses are quoted by every
commentary on that work. So text that recurs across books is disproportionately
likely to be root text — and unlike a classifier, a match carries its own
boundaries, which is exactly what the current model gets wrong.

This measures the signal directly. It marks every character of a book that also
appears in some other book, then compares that mask against the gold tsawa
annotation:

    precision = of characters flagged as repeated, how many are gold tsawa
    recall    = of gold tsawa characters, how many were flagged

A precision well above the 4.8% base rate means repetition is informative. High
recall as well would mean a retrieval-first system is viable on its own; high
precision with low recall means it belongs as an auxiliary signal to the model.

Method: shingle every book into overlapping fixed-length substrings, hash them,
and note which books each hash appears in. Then for the books under test, mark
characters covered by a shingle seen elsewhere.

Usage
-----
    python test_repetition_signal.py --books 80 --shingle 60 --stride 15
"""

from __future__ import annotations

import argparse
import hashlib
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def norm_map(text: str):
    """Normalised text plus an index back to original character positions."""
    out, idx = [], []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        out.append(unicodedata.normalize("NFC", ch))
        idx.append(i)
    return "".join(out), idx


def h(s: str) -> bytes:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="data/processed/tsawa/tsawa_spans_resolved.csv")
    ap.add_argument("--raw-opf", default="data/raw_opf")
    ap.add_argument("--books", type=int, default=80,
                    help="books to index (all of them is slow and memory-hungry)")
    ap.add_argument("--test-books", type=int, default=15,
                    help="of those, how many to evaluate on")
    ap.add_argument("--shingle", type=int, default=60,
                    help="substring length; shorter matches more but by chance")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="scratch/tsawa/analysis/repetition_signal.csv")
    args = ap.parse_args()

    random.seed(args.seed)
    raw = Path(args.raw_opf)

    d = pd.read_csv(args.spans)
    if "dropped" in d.columns:
        d = d[~d.dropped.astype(bool)]
    annotated = set(d.pecha_id.unique())

    # index books that have a Tsawa layer, so the test books have gold to compare
    pids = sorted(annotated)
    random.shuffle(pids)
    pids = pids[:args.books]

    texts, nmaps = {}, {}
    for pid in pids:
        for cand in (raw / f"{pid}.opf" / f"{pid}.opf" / "base" / "v001.txt",
                     raw / f"{pid}.opf" / "base" / "v001.txt"):
            if cand.is_file():
                t = cand.read_text(encoding="utf-8")
                texts[pid] = t
                nmaps[pid] = norm_map(t)
                break
    pids = [p for p in pids if p in texts]
    print(f"indexing {len(pids)} books "
          f"({sum(len(t) for t in texts.values()):,} chars)")

    # ---- build the shingle index ----------------------------------------
    seen: dict[bytes, set[str]] = defaultdict(set)
    for n, pid in enumerate(pids, 1):
        nt, _ = nmaps[pid]
        for i in range(0, max(0, len(nt) - args.shingle), args.stride):
            seen[h(nt[i:i + args.shingle])].add(pid)
        if n % 20 == 0:
            print(f"  {n}/{len(pids)} indexed, {len(seen):,} shingles", flush=True)

    shared = {k for k, v in seen.items() if len(v) > 1}
    print(f"  {len(shared):,} shingles appear in more than one book "
          f"({100*len(shared)/max(len(seen),1):.2f}%)")

    # ---- evaluate on a few books ----------------------------------------
    test = pids[:args.test_books]
    rows = []
    TP = FP = FN = TN = 0

    for pid in test:
        text = texts[pid]
        nt, idx = nmaps[pid]

        gold = np.zeros(len(text), dtype=bool)
        for _, r in d[d.pecha_id == pid].iterrows():
            gold[int(r.start):int(r.end)] = True

        flag = np.zeros(len(text), dtype=bool)
        for i in range(0, max(0, len(nt) - args.shingle), args.stride):
            key = h(nt[i:i + args.shingle])
            others = seen.get(key, set()) - {pid}
            if others:
                lo = idx[i]
                hi = idx[min(i + args.shingle - 1, len(idx) - 1)]
                flag[lo:hi + 1] = True

        tp = int((flag & gold).sum())
        fp = int((flag & ~gold).sum())
        fn = int((~flag & gold).sum())
        tn = int((~flag & ~gold).sum())
        TP += tp; FP += fp; FN += fn; TN += tn

        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"pecha_id": pid, "chars": len(text),
                     "gold_pct": 100 * gold.mean(),
                     "flagged_pct": 100 * flag.mean(),
                     "precision": p, "recall": r_})
        print(f"  {pid}: gold {100*gold.mean():5.2f}%  flagged "
              f"{100*flag.mean():5.2f}%  P={p:.3f} R={r_:.3f}")

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    base = (TP + FN) / (TP + FP + FN + TN)

    print(f"\n{'='*64}")
    print("REPETITION AS A TSAWA SIGNAL (character level)")
    print("=" * 64)
    print(f"  precision {prec:.4f}   recall {rec:.4f}   F1 {f1:.4f}")
    print(f"  base rate (tsawa share of all characters): {base:.4f}")
    if base:
        print(f"  lift over base rate: {prec/base:.1f}x")

    print()
    if prec > base * 4 and rec > 0.3:
        print("  STRONG. Repetition alone locates a large share of tsawa with")
        print("  far better precision than chance. Worth building as a")
        print("  retrieval stage feeding the model, or as a prior on its logits.")
    elif prec > base * 2:
        print("  USEFUL BUT PARTIAL. Better than chance, not sufficient alone.")
        print("  Best used as an extra input feature or a confidence boost on")
        print("  spans the model already proposes.")
    else:
        print("  WEAK. Repeated text is not specifically tsawa here — shared")
        print("  phrasing, formulae and stock passages recur too. Drop it.")

    print("\n  Caveats: only these books are indexed, so a verse whose other")
    print("  witnesses are outside the sample looks unique. Recall is a floor.")
    print("  Shorter shingles would match more and mean less.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()