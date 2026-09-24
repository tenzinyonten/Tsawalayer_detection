#!/usr/bin/env python3
"""
span_metrics.py — score the same predictions under every metric definition
that "50% match" could mean.

"50% match" is ambiguous, and for this model the ambiguity is worth a large
number of F1 points:

  IoU >= 0.5          a prediction 2x too long FAILS
  gold coverage >= 0.5 a prediction 10x too long PASSES at 100%

31.5% of gold spans are swallowed by oversized predictions, so those two
definitions disagree sharply here. Rather than guess which one the team uses,
compute both and read off whichever they meant.

Input is the CSV written by show_missed_spans.py, which already holds each
gold span, its verdict, and the size of the prediction that overlapped it.

Usage
-----
    python span_metrics.py --csv scratch/tsawa/analysis/missed_v2_io_val_all.csv
    python span_metrics.py --csv scratch/tsawa/analysis/missed_v1_bio_v2val.csv --label "v1 BIO"
"""

from __future__ import annotations

import argparse

import pandas as pd


def prf(tp: float, fp: float, fn: float):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    name = args.label or args.csv
    n_gold = len(df)

    print(f"\n{'='*66}")
    print(f"{name}   —   {n_gold:,} gold spans")
    print("=" * 66)

    # ---- verdict distribution -------------------------------------------
    print("\nfailure taxonomy (MUC-5 names in brackets):")
    muc = {"MATCHED": "Correct", "CONTAINED": "Partial / over-inclusive",
           "OVERLAP": "Partial", "SPLIT": "Partial + Spurious",
           "MISSED": "Missing"}
    for v in ("MATCHED", "CONTAINED", "OVERLAP", "SPLIT", "MISSED"):
        n = (df.verdict == v).sum()
        print(f"  {v:>10}: {n:>6,} ({100*n/n_gold:5.1f}%)   [{muc[v]}]")

    # ---- reconstruct per-span overlap ------------------------------------
    # The CSV records gold length and the length of the overlapping prediction.
    # For CONTAINED the prediction fully covers the gold span, so
    # intersection == gold length. That is the case the metric choice hinges on,
    # and it is the only one we can score exactly from these columns.
    has_pred = df.n_overlapping_preds > 0
    gold_len = df.gold_len.astype(float)
    pred_len = df.containing_pred_len.astype(float)

    contained = df.verdict == "CONTAINED"
    matched = df.verdict == "MATCHED"

    # IoU for contained spans: |G| / |P|   (since G is entirely inside P)
    iou = pd.Series(0.0, index=df.index)
    iou[contained] = gold_len[contained] / pred_len[contained].clip(lower=1)
    iou[matched] = 1.0  # within tolerance; treat as exact for this purpose

    # Gold coverage: fraction of the gold span covered.
    cover = pd.Series(0.0, index=df.index)
    cover[contained] = 1.0        # fully swallowed == fully covered
    cover[matched] = 1.0

    print("\nNOTE: OVERLAP and SPLIT spans are scored as 0 below — the CSV does")
    print("not record their exact intersection, so these numbers are a LOWER")
    print(f"BOUND. {(df.verdict.isin(['OVERLAP','SPLIT'])).sum():,} spans "
          f"({100*(df.verdict.isin(['OVERLAP','SPLIT'])).sum()/n_gold:.1f}%) "
          "are affected.")

    # ---- the two families -------------------------------------------------
    print("\n" + "-" * 66)
    print("IoU THRESHOLD  (symmetric — punishes oversized predictions)")
    print("-" * 66)
    for thr in (0.5, 0.7, 0.9, 1.0):
        tp = int((iou >= thr).sum())
        fn = n_gold - tp
        # every gold span the model failed to match leaves its prediction
        # unmatched too; without the full prediction list this is approximate
        fp = int(has_pred.sum() - tp)
        p, r, f = prf(tp, fp, fn)
        print(f"  IoU >= {thr:.1f}:  P={p:.4f}  R={r:.4f}  F1={f:.4f}   (TP={tp:,})")

    print("\n" + "-" * 66)
    print("GOLD COVERAGE  (asymmetric — an oversized prediction still passes)")
    print("-" * 66)
    for thr in (0.5, 0.7, 1.0):
        tp = int((cover >= thr).sum())
        fn = n_gold - tp
        fp = int(has_pred.sum() - tp)
        p, r, f = prf(tp, fp, fn)
        print(f"  cover >= {thr:.1f}:  P={p:.4f}  R={r:.4f}  F1={f:.4f}   (TP={tp:,})")

    # ---- why it matters ---------------------------------------------------
    n_cont = int(contained.sum())
    if n_cont:
        med = (gold_len[contained] / pred_len[contained].clip(lower=1)).median()
        print(f"\n{'='*66}")
        print("WHY THE DEFINITION MATTERS HERE")
        print("=" * 66)
        print(f"  {n_cont:,} gold spans ({100*n_cont/n_gold:.1f}%) are swallowed "
              f"by an oversized prediction.")
        print(f"  Median IoU for those: {med:.3f}")
        print(f"  Under gold coverage they all count as CORRECT.")
        print(f"  Under IoU>=0.5 only {int((iou[contained]>=0.5).sum()):,} of them do.")
        print(f"\n  So the same model scores very differently depending on which")
        print(f"  '50% match' the team means. Ask before reporting a number.")


if __name__ == "__main__":
    main()