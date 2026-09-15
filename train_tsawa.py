#!/usr/bin/env python3
"""
train_tsawa.py (v3) — fine-tune mmBERT-base as a binary Tsawa (root-text) tagger.

Two tagging schemes, selectable at load time — no dataset rebuild needed:

  BIO (default)  O=0, B-TSAWA=1, I-TSAWA=2
  IO  (--io)     O=0, TSAWA=1     (B collapsed into I)

Why IO is worth trying: B is 0.062% of tokens, and v1 error analysis showed the
model emitting spurious B tags inside spans it had already found correctly
(one gold span predicted as 3-80 pieces). IO removes the B class entirely and
recovers span starts from 0->1 transitions, so fragmentation becomes
structurally impossible rather than something to be suppressed with loss
weights.

v1 baseline (v1 split, BIO, sqrt_inv weights, no warmup):
  test soft_f1_tol1 0.177 (training-time, bf16) / 0.091 (offline, fp32)
  52% of gold spans missed entirely; 0.000 recall on spans under 10 tokens;
  23.6% of found spans fragmented; over-prediction 1.8x-3.1x
These are NOT comparable to v2 runs — different test split.

The test split is FROZEN. Use --skip-test while tuning.

Examples
--------
    # 1. baseline on the v2 split: unweighted BIO
    python train_tsawa.py --weight-scheme none --skip-test \
        --output-dir /workspace/runs/v2_bio_none

    # 2. the real intervention: IO tagging
    python train_tsawa.py --io --weight-scheme none --skip-test \
        --output-dir /workspace/runs/v2_io_none

Set REPORT_TO=wandb and WANDB_PROJECT=tsawa for wandb logging.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

O, B, I = 0, 1, 2
BIO_NAMES = ["O", "B-TSAWA", "I-TSAWA"]
IO_NAMES = ["O", "TSAWA"]

# Length buckets in TOKENS, for the per-bucket recall breakdown. The first two
# are where v1 scored exactly zero.
BUCKETS = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 300), (301, 10**9)]


# ---------------------------------------------------------------------------
# label handling
# ---------------------------------------------------------------------------

def to_io(labels: np.ndarray) -> np.ndarray:
    """Collapse BIO -> IO: B(1) and I(2) both become 1. -100 padding preserved."""
    out = labels.copy()
    out[out == I] = 1
    return out


def count_labels(ds, n_labels: int, io: bool) -> np.ndarray:
    counts = np.zeros(n_labels, dtype=np.int64)
    for batch in ds.with_format("numpy").iter(batch_size=64):
        lab = batch["labels"].reshape(-1)
        lab = lab[lab != -100]
        if io:
            lab = to_io(lab)
        counts += np.bincount(lab, minlength=n_labels)
    return counts


def compute_weights(counts: np.ndarray, scheme: str, cap: float,
                    manual: str | None, io: bool):
    n = len(counts)
    n_o = float(counts[O])

    if scheme == "none":
        w = np.ones(n, dtype=np.float64)
    elif scheme == "legacy":
        if io:
            raise SystemExit("--weight-scheme legacy is BIO-only (it sets a B weight)")
        w = np.array([0.07, 5.0, 5.0], dtype=np.float64)
        cap = 0.0
    elif scheme in ("inv", "sqrt_inv"):
        ratios = np.array([1.0] + [n_o / counts[c] for c in range(1, n)])
        w = np.sqrt(ratios) if scheme == "sqrt_inv" else ratios
        w[O] = 1.0
    elif scheme == "manual":
        if not manual:
            raise SystemExit("--weight-scheme manual requires --manual-weights")
        w = np.array([float(x) for x in manual.split(",")], dtype=np.float64)
        if w.shape != (n,):
            raise SystemExit(f"--manual-weights needs exactly {n} values for this scheme")
    else:
        raise SystemExit(f"unknown weight scheme: {scheme}")

    raw = w.copy()
    if cap and cap > 0:
        w = np.minimum(w, cap)
    return w, raw


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------

def extract_spans_bio(seq: np.ndarray):
    spans, start = [], None
    for i, lab in enumerate(seq):
        if lab == B:
            if start is not None:
                spans.append((start, i))
            start = i
        elif lab == I:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(seq)))
    return spans


def extract_spans_io(seq: np.ndarray):
    """Contiguous runs of 1. Starts come from 0->1 transitions."""
    spans, start = [], None
    for i, lab in enumerate(seq):
        if lab == 1 and start is None:
            start = i
        elif lab != 1 and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(seq)))
    return spans


def match(gold, pred, tol: int):
    """Greedy one-to-one; both boundaries within tol."""
    remaining = list(pred)
    tp = 0
    for g0, g1 in gold:
        for k, (p0, p1) in enumerate(remaining):
            if abs(p0 - g0) <= tol and abs(p1 - g1) <= tol:
                tp += 1
                remaining.pop(k)
                break
    return tp, len(pred) - tp, len(gold) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bucket_of(n):
    for b in BUCKETS:
        if b[0] <= n <= b[1]:
            return b
    return BUCKETS[-1]


def build_metrics(tol: int, io: bool, names: list[str]):
    extract = extract_spans_io if io else extract_spans_bio
    n_labels = len(names)

    def compute_metrics(eval_pred):
        preds = np.asarray(eval_pred[0])
        labels = np.asarray(eval_pred[1])

        TP = FP = FN = 0
        TP0 = FP0 = FN0 = 0
        tok_correct = tok_total = 0
        cls_tp = np.zeros(n_labels)
        cls_fp = np.zeros(n_labels)
        cls_fn = np.zeros(n_labels)

        orphan_preds = 0
        frag_gold = 0
        missed_gold = 0
        bucket_gold = {b: 0 for b in BUCKETS}
        bucket_hit = {b: 0 for b in BUCKETS}

        for p_row, l_row in zip(preds, labels):
            m = l_row != -100
            p, l = p_row[m], l_row[m]
            tok_correct += int((p == l).sum())
            tok_total += int(m.sum())
            for c in range(n_labels):
                cls_tp[c] += int(((p == c) & (l == c)).sum())
                cls_fp[c] += int(((p == c) & (l != c)).sum())
                cls_fn[c] += int(((p != c) & (l == c)).sum())

            g, pr = extract(l), extract(p)
            tp, fp, fn = match(g, pr, tol)
            TP += tp; FP += fp; FN += fn
            tp0, fp0, fn0 = match(g, pr, 0)
            TP0 += tp0; FP0 += fp0; FN0 += fn0

            # structural diagnostics
            for gs in g:
                hits = [ps for ps in pr
                        if min(gs[1], ps[1]) - max(gs[0], ps[0]) > 0]
                if not hits:
                    missed_gold += 1
                elif len(hits) >= 2:
                    frag_gold += 1
                bk = bucket_of(gs[1] - gs[0])
                bucket_gold[bk] += 1
                if any(abs(ps[0] - gs[0]) <= tol and abs(ps[1] - gs[1]) <= tol
                       for ps in pr):
                    bucket_hit[bk] += 1
            for ps in pr:
                if not any(min(gs[1], ps[1]) - max(gs[0], ps[0]) > 0 for gs in g):
                    orphan_preds += 1

        prec, rec, f1 = prf(TP, FP, FN)
        _, _, f1_exact = prf(TP0, FP0, FN0)
        n_gold, n_pred = TP + FN, TP + FP

        out = {
            f"soft_f1_tol{tol}": f1,
            f"soft_precision_tol{tol}": prec,
            f"soft_recall_tol{tol}": rec,
            "exact_span_f1": f1_exact,
            "token_accuracy": tok_correct / tok_total if tok_total else 0.0,
            "gold_spans": int(n_gold),
            "pred_spans": int(n_pred),
            "over_prediction_ratio": n_pred / n_gold if n_gold else 0.0,
            # v1 reference values, for orientation:
            "missed_gold_pct": 100 * missed_gold / n_gold if n_gold else 0.0,   # v1: 52.0
            "fragmented_gold_pct": 100 * frag_gold / n_gold if n_gold else 0.0, # v1: 23.6
            "orphan_pred_pct": 100 * orphan_preds / n_pred if n_pred else 0.0,  # v1: 46.7
        }
        for c, name in enumerate(names):
            out[f"f1_{name}"] = prf(cls_tp[c], cls_fp[c], cls_fn[c])[2]
        for b in BUCKETS:
            if bucket_gold[b]:
                tag = f"{b[0]}-{b[1]}" if b[1] < 10**9 else f"{b[0]}plus"
                out[f"recall_tok_{tag}"] = bucket_hit[b] / bucket_gold[b]
        return out
    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device), ignore_index=-100
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Yontenn/formatting-tsawa-v2")
    ap.add_argument("--base-model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--output-dir", default="/workspace/runs/tsawa")

    ap.add_argument("--io", action="store_true",
                    help="IO tagging: collapse B into I, 2 labels instead of 3")
    ap.add_argument("--weight-scheme", default="none",
                    choices=["none", "sqrt_inv", "inv", "legacy", "manual"])
    ap.add_argument("--weight-cap", type=float, default=0.0)
    ap.add_argument("--manual-weights", default=None)

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=0.3)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="fallback if warmup_ratio is unsupported; 0 = derive from ratio")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)

    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--attn", default="auto",
                    choices=["auto", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-test", action="store_true",
                    help="leave the frozen test split alone while tuning")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = IO_NAMES if args.io else BIO_NAMES
    n_labels = len(names)
    print(f"tagging scheme: {'IO' if args.io else 'BIO'} ({n_labels} labels)")

    # ---- data -------------------------------------------------------------
    ds = (load_from_disk(args.dataset) if Path(args.dataset).exists()
          else load_dataset(args.dataset, token=os.environ.get("HF_TOKEN")))
    print({k: len(v) for k, v in ds.items()})

    keep = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    if args.io:
        # Collapse at load time; the stored dataset stays BIO.
        ds = ds.map(
            lambda b: {"labels": [[1 if x == I else x for x in row]
                                  for row in b["labels"]]},
            batched=True, batch_size=64, desc="BIO -> IO",
        )
    ds.set_format("torch")

    train_ds, eval_ds = ds["train"], ds["validation"]
    if args.smoke_test:
        train_ds = train_ds.select(range(min(32, len(train_ds))))
        eval_ds = eval_ds.select(range(min(8, len(eval_ds))))
        args.epochs = 1

    # ---- class weights ----------------------------------------------------
    counts = count_labels(ds["train"], n_labels, io=False)  # already collapsed
    total = counts.sum()
    print("\ntrain label counts (padding excluded):")
    for c, name in enumerate(names):
        print(f"  {name:>8}: {counts[c]:>12,}  ({100*counts[c]/total:6.3f}%)")
    pos = total - counts[O]
    print(f"  positive: {100*pos/total:.3f}%")

    w, raw = compute_weights(counts, args.weight_scheme, args.weight_cap,
                             args.manual_weights, args.io)
    print(f"\nweight scheme: {args.weight_scheme}  cap: {args.weight_cap or 'none'}")
    for c, name in enumerate(names):
        note = "  <-- CLIPPED" if raw[c] != w[c] else ""
        print(f"  {name:>8}: raw {raw[c]:>10.3f} -> used {w[c]:>8.3f}{note}")
    class_weights = torch.tensor(w, dtype=torch.float32)

    # ---- model ------------------------------------------------------------
    cfg = AutoConfig.from_pretrained(
        args.base_model, num_labels=n_labels,
        id2label={i: n for i, n in enumerate(names)},
        label2id={n: i for i, n in enumerate(names)},
    )
    attn = args.attn
    if attn == "auto":
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
    print(f"attention implementation: {attn}")
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            args.base_model, config=cfg, attn_implementation=attn)
    except Exception as e:
        print(f"  {attn} failed ({e}); falling back to sdpa")
        model = AutoModelForTokenClassification.from_pretrained(
            args.base_model, config=cfg, attn_implementation="sdpa")

    # ---- training args ----------------------------------------------------
    sig = inspect.signature(TrainingArguments.__init__).parameters

    ta_kwargs = dict(
        output_dir=str(out),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        bf16=not args.no_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=not args.no_grad_checkpointing,
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_soft_f1_tol{args.tol}",
        greater_is_better=True,
        report_to=os.environ.get("REPORT_TO", "none"),
        run_name=args.run_name or
                 f"tsawa-v2-{'io' if args.io else 'bio'}-{args.weight_scheme}",
        seed=args.seed,
        dataloader_num_workers=2,
    )
    ta_kwargs["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = "epoch"
    ta_kwargs["save_strategy"] = "epoch"

    # Warmup fallback. v1 lost warmup entirely because transformers 5.x renamed
    # the arg and the old code filtered it out without saying so.
    if "warmup_ratio" not in sig:
        steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
        derived = args.warmup_steps or int(steps_per_epoch * args.epochs * args.warmup_ratio)
        if "warmup_steps" in sig:
            print(f"\n  warmup_ratio unsupported; using warmup_steps={derived}")
            ta_kwargs.pop("warmup_ratio")
            ta_kwargs["warmup_steps"] = derived
        else:
            print("\n  *** WARNING: no warmup parameter available. Training will "
                  "start at full LR. Consider --lr 5e-6. ***")

    dropped = [k for k in ta_kwargs if k not in sig]
    if dropped:
        print(f"  dropping unsupported TrainingArguments: {dropped}")
        ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in sig}

    targs = TrainingArguments(**ta_kwargs)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=build_metrics(args.tol, args.io, names),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train(resume_from_checkpoint=args.resume or None)

    # ---- evaluate ---------------------------------------------------------
    val = trainer.evaluate(eval_ds, metric_key_prefix="val")
    print("\nVALIDATION:", json.dumps(val, indent=2, default=float))

    test = None
    if args.skip_test:
        print("\nTEST: skipped (--skip-test). Frozen split, final run only.")
    else:
        test = trainer.evaluate(ds["test"], metric_key_prefix="test")
        print("\nTEST:", json.dumps(test, indent=2, default=float))

    print("\nCompare against v1 (different split, orientation only):")
    print("  missed_gold 52.0% | fragmented_gold 23.6% | orphan_pred 46.7%")
    print("  recall was 0.000 on spans under 10 tokens")
    print("Watch recall_tok_1-5 and recall_tok_6-10: if still zero, short spans")
    print("are unlearnable as labelled, not merely underweighted.")

    trainer.save_model(str(out / "best"))
    (out / "results.json").write_text(json.dumps({
        "args": vars(args),
        "tagging": "IO" if args.io else "BIO",
        "split": "v2 (split_v2_frozen.csv)",
        "label_counts": counts.tolist(),
        "weights_raw": raw.tolist(),
        "weights_used": w.tolist(),
        "validation": {k: float(v) for k, v in val.items() if isinstance(v, (int, float))},
        "test": ({k: float(v) for k, v in test.items() if isinstance(v, (int, float))}
                 if test else None),
    }, indent=2))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()