#!/usr/bin/env python3
"""
eval_by_batch.py — is the gap to the joint model a batch problem?

The joint model's tsawa figure (0.395 IoU@0.5) comes from karma689/layer_detection,
which is OLD batch only (266 P-prefix books). This project trains and scores on
old + new combined. The new batch differs on almost everything measured:
46% short spans vs 0.47%, all five interlinear books, the nested quotations,
different layer names, the opposite direction of boundary bug.

So 0.323 here and 0.395 there may not be the same task. This scores one model on
validation, split by source batch. If old-batch validation lands near 0.4 and
new-batch far below, the gap is the data, not the pipeline.

Reuses the decoder and metric from eval_particle_rule.py — keep both files in
the same folder.

Usage
-----
    python eval_by_batch.py --model Yontenn/mmbert-tsawa-bio-inv-v2 \\
        --dataset ds_v2 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForTokenClassification

from eval_particle_rule import score, spans, viterbi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Yontenn/mmbert-tsawa-bio-inv-v2")
    ap.add_argument("--dataset", default="ds_v2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--break-penalty", type=float, default=12.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    ds = (load_from_disk(args.dataset) if Path(args.dataset).exists()
          else load_dataset(args.dataset, token=os.environ.get("HF_TOKEN")))
    split = ds[args.split]

    # batch label per window: prefer the stored column, else derive from pecha_id
    if "source_batch" in split.column_names:
        batches = [str(x) for x in split["source_batch"]]
    elif "pecha_id" in split.column_names:
        batches = ["old" if str(p).startswith("P") else "new"
                   for p in split["pecha_id"]]
    else:
        raise SystemExit(f"no batch or pecha column in {split.column_names}")
    pechas = split["pecha_id"] if "pecha_id" in split.column_names else [""] * len(split)

    model = AutoModelForTokenClassification.from_pretrained(args.model)
    use_feat = "features" in split.column_names
    if use_feat:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "tsawa"))
        from tsawa_feat_model import TsawaFeatModel
        try:
            model = TsawaFeatModel.from_pretrained(args.model)
        except Exception:
            pass
    model.eval().to(args.device)

    keep = ["input_ids", "attention_mask", "labels"]
    if use_feat:
        keep.append("features")
    t = split.remove_columns([c for c in split.column_names if c not in keep])
    t.set_format("torch")
    print(f"{args.split}: {len(t)} windows — "
          + ", ".join(f"{k}: {v}" for k, v in
                      sorted(defaultdict(int, {b: batches.count(b) for b in set(batches)}).items())))

    gold = defaultdict(list)
    pred = defaultdict(list)
    books = defaultdict(set)
    with torch.no_grad():
        for i in range(0, len(t), args.batch_size):
            b = t[i:i + args.batch_size]
            kw = dict(input_ids=b["input_ids"].to(args.device),
                      attention_mask=b["attention_mask"].to(args.device))
            if use_feat:
                kw["features"] = b["features"].to(args.device)
            lg = model(**kw).logits.float().cpu().numpy()
            for j in range(len(lg)):
                w = i + j
                m = b["labels"][j].numpy() != -100
                g = spans(b["labels"][j].numpy()[m])
                p = spans(viterbi(lg[j][m], args.break_penalty))
                for key in (batches[w], "ALL"):
                    gold[key].append(g)
                    pred[key].append(p)
                books[batches[w]].add(pechas[w])
            if (i // args.batch_size) % 25 == 0:
                print(f"  {i}/{len(t)}", flush=True)

    print(f"\n{'='*64}")
    print(f"IoU@0.5 BY BATCH — {args.model}")
    print("=" * 64)
    print(f"  {'batch':<8}{'books':>7}{'gold':>8}{'IoU@0.5':>10}{'P':>8}{'R':>8}")
    for key in sorted(k for k in gold if k != "ALL") + ["ALL"]:
        f1, p, r, _ = score(gold[key], pred[key])
        n = sum(len(x) for x in gold[key])
        nb = len(books[key]) if key != "ALL" else sum(len(v) for v in books.values())
        print(f"  {key:<8}{nb:>7}{n:>8,}{f1:>10.4f}{p:>8.3f}{r:>8.3f}")

    print("\n  The joint model's 0.395 was measured on OLD-batch books only.")
    print("  Compare it against the old row, not the ALL row.")
    print("\n  If old is near or above 0.395 and new is far lower, the gap to")
    print("  the joint model is the new batch's annotation, not this pipeline.")
    print("  If both rows are similar, batch is not the explanation.")


if __name__ == "__main__":
    main()