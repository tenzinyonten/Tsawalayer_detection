#!/usr/bin/env python3
"""
train_tsawa.py — fine-tune mmBERT-base as a binary Tsawa (root-text) tagger.

Labels: O=0, B-TSAWA=1, I-TSAWA=2, padding=-100.

The class weighting is a flag, not a constant, so the inherited joint-model
config and the recomputed one are one run each:

    --weight-scheme legacy      # O=0.07, B=I=5.0  (inherited from joint model)
    --weight-scheme sqrt_inv    # O=1.0,  B~39.2, I~4.5  (recomputed, recommended)

Selection metric is span-level soft F1 with +/-1 token boundary tolerance.

Examples
--------
    # recommended first run
    python train_tsawa.py --dataset Yontenn/formatting-tsawa-v1 \
        --weight-scheme sqrt_inv --output-dir /workspace/runs/sqrt_inv

    # the ablation
    python train_tsawa.py --dataset Yontenn/formatting-tsawa-v1 \
        --weight-scheme legacy --output-dir /workspace/runs/legacy

    # smoke test before burning GPU hours
    python train_tsawa.py --dataset ./data/processed/tsawa_dataset --smoke-test
"""

from __future__ import annotations

import argparse
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
LABEL_NAMES = ["O", "B-TSAWA", "I-TSAWA"]


# ---------------------------------------------------------------------------
# class weights
# ---------------------------------------------------------------------------

def count_labels(ds) -> np.ndarray:
    counts = np.zeros(3, dtype=np.int64)
    for batch in ds.with_format("numpy").iter(batch_size=64):
        lab = batch["labels"].reshape(-1)
        lab = lab[lab != -100]
        counts += np.bincount(lab, minlength=3)
    return counts


def compute_weights(counts: np.ndarray, scheme: str, cap: float,
                    manual: str | None) -> np.ndarray:
    n_o = float(counts[O])
    if scheme == "none":
        w = np.ones(3, dtype=np.float64)
    elif scheme == "legacy":
        # Inherited from the joint multi-label model. Note both B and I clip to
        # the same 5.0 despite B being ~75x rarer than I — that flattening is
        # the thing this scheme is here to be compared against.
        w = np.array([0.07, 5.0, 5.0], dtype=np.float64)
        cap = 0.0  # already baked in; don't re-clip
    elif scheme == "inv":
        w = np.array([1.0, n_o / counts[B], n_o / counts[I]])
    elif scheme == "sqrt_inv":
        w = np.array([1.0, np.sqrt(n_o / counts[B]), np.sqrt(n_o / counts[I])])
    elif scheme == "manual":
        if not manual:
            raise SystemExit("--weight-scheme manual requires --manual-weights")
        w = np.array([float(x) for x in manual.split(",")], dtype=np.float64)
        if w.shape != (3,):
            raise SystemExit("--manual-weights needs exactly 3 comma-separated values")
    else:
        raise SystemExit(f"unknown weight scheme: {scheme}")

    raw = w.copy()
    if cap and cap > 0:
        w = np.minimum(w, cap)
    return w, raw


# ---------------------------------------------------------------------------
# span-level soft F1
# ---------------------------------------------------------------------------

def extract_spans(seq: np.ndarray) -> list[tuple[int, int]]:
    """BIO sequence -> list of (start, end_exclusive) token spans."""
    spans, start = [], None
    for i, lab in enumerate(seq):
        if lab == B:
            if start is not None:
                spans.append((start, i))
            start = i
        elif lab == I:
            if start is None:
                start = i  # tolerate I without B
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(seq)))
    return spans


def soft_f1(gold: list[tuple[int, int]], pred: list[tuple[int, int]],
            tol: int) -> tuple[int, int, int]:
    """Greedy one-to-one match where both boundaries are within `tol` tokens."""
    unmatched = list(pred)
    tp = 0
    for g0, g1 in gold:
        for k, (p0, p1) in enumerate(unmatched):
            if abs(p0 - g0) <= tol and abs(p1 - g1) <= tol:
                tp += 1
                unmatched.pop(k)
                break
    return tp, len(pred) - tp, len(gold) - tp  # tp, fp, fn


def build_metrics(tol: int):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds)
        labels = np.asarray(labels)

        TP = FP = FN = 0
        tok_correct = tok_total = 0
        per_class_tp = np.zeros(3)
        per_class_fp = np.zeros(3)
        per_class_fn = np.zeros(3)

        for p_row, l_row in zip(preds, labels):
            mask = l_row != -100
            p, l = p_row[mask], l_row[mask]
            tok_correct += int((p == l).sum())
            tok_total += int(mask.sum())
            for c in (O, B, I):
                per_class_tp[c] += int(((p == c) & (l == c)).sum())
                per_class_fp[c] += int(((p == c) & (l != c)).sum())
                per_class_fn[c] += int(((p != c) & (l == c)).sum())
            tp, fp, fn = soft_f1(extract_spans(l), extract_spans(p), tol)
            TP += tp; FP += fp; FN += fn

        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

        out = {
            f"soft_f1_tol{tol}": f1,
            f"soft_precision_tol{tol}": prec,
            f"soft_recall_tol{tol}": rec,
            "token_accuracy": tok_correct / tok_total if tok_total else 0.0,
            "gold_spans": int(TP + FN),
            "pred_spans": int(TP + FP),
        }
        for c, name in enumerate(LABEL_NAMES):
            p_ = per_class_tp[c] / (per_class_tp[c] + per_class_fp[c]) if per_class_tp[c] + per_class_fp[c] else 0.0
            r_ = per_class_tp[c] / (per_class_tp[c] + per_class_fn[c]) if per_class_tp[c] + per_class_fn[c] else 0.0
            out[f"f1_{name}"] = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        return out
    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """Argmax on GPU so the Trainer doesn't accumulate full logits in RAM."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# weighted trainer
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
    ap.add_argument("--dataset", default="Yontenn/formatting-tsawa-v1",
                    help="HF repo id or local path from save_to_disk")
    ap.add_argument("--base-model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--output-dir", default="/workspace/runs/tsawa")

    ap.add_argument("--weight-scheme", default="sqrt_inv",
                    choices=["sqrt_inv", "inv", "legacy", "none", "manual"])
    ap.add_argument("--weight-cap", type=float, default=0.0,
                    help="0 = no cap. The inherited 5.0 binds on every scheme.")
    ap.add_argument("--manual-weights", default=None, help="e.g. 1.0,39.2,4.5")

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=0.3)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--attn", default="auto",
                    choices=["auto", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-test", action="store_true",
                    help="32 train / 8 eval windows, 1 epoch — checks it runs")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data -------------------------------------------------------------
    if Path(args.dataset).exists():
        ds = load_from_disk(args.dataset)
    else:
        ds = load_dataset(args.dataset, token=os.environ.get("HF_TOKEN"))
    print({k: len(v) for k, v in ds.items()})

    keep = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    ds.set_format("torch")

    train_ds, eval_ds = ds["train"], ds["validation"]
    if args.smoke_test:
        train_ds = train_ds.select(range(min(32, len(train_ds))))
        eval_ds = eval_ds.select(range(min(8, len(eval_ds))))
        args.epochs = 1

    # ---- class weights ----------------------------------------------------
    counts = count_labels(ds["train"])
    total = counts.sum()
    print("\ntrain label counts (padding excluded):")
    for c, name in enumerate(LABEL_NAMES):
        print(f"  {name:>8}: {counts[c]:>12,}  ({100*counts[c]/total:6.3f}%)")
    print(f"  positive (B+I): {100*(counts[B]+counts[I])/total:.3f}%")

    w, raw = compute_weights(counts, args.weight_scheme, args.weight_cap,
                             args.manual_weights)
    print(f"\nweight scheme: {args.weight_scheme}  cap: {args.weight_cap or 'none'}")
    for c, name in enumerate(LABEL_NAMES):
        note = "  <-- CLIPPED" if raw[c] != w[c] else ""
        print(f"  {name:>8}: raw {raw[c]:>10.3f} -> used {w[c]:>8.3f}{note}")
    if args.weight_cap and (raw > args.weight_cap).any():
        print("  NOTE: the cap is binding. Weights are set by the cap, not the scheme.")
    class_weights = torch.tensor(w, dtype=torch.float32)

    # ---- model ------------------------------------------------------------
    cfg = AutoConfig.from_pretrained(
        args.base_model, num_labels=3,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
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
        metric_for_best_model=f"soft_f1_tol{args.tol}",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )
    # transformers renamed evaluation_strategy -> eval_strategy in 4.46
    import inspect
    sig = inspect.signature(TrainingArguments.__init__).parameters
    key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    ta_kwargs[key] = "epoch"
    ta_kwargs["save_strategy"] = "epoch"
    targs = TrainingArguments(**ta_kwargs)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=build_metrics(args.tol),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train(resume_from_checkpoint=args.resume or None)

    # ---- evaluate ---------------------------------------------------------
    val = trainer.evaluate(eval_ds, metric_key_prefix="val")
    test = trainer.evaluate(ds["test"], metric_key_prefix="test")
    print("\nVALIDATION:", json.dumps(val, indent=2, default=float))
    print("\nTEST:", json.dumps(test, indent=2, default=float))
    print("\nNOTE: val positive density is ~4.17% and test ~5.53%, so val and "
          "test soft F1 are not directly comparable. Compare runs on the same split.")

    trainer.save_model(str(out / "best"))
    (out / "results.json").write_text(json.dumps({
        "args": vars(args),
        "label_counts": counts.tolist(),
        "weights_raw": raw.tolist(),
        "weights_used": w.tolist(),
        "validation": {k: float(v) for k, v in val.items() if isinstance(v, (int, float))},
        "test": {k: float(v) for k, v in test.items() if isinstance(v, (int, float))},
    }, indent=2))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
