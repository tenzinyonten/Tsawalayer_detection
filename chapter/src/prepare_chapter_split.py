#!/usr/bin/env python3
"""Frozen book-level split for the Chapter dataset.

* Books: every book with at least one cleaned Chapter span, minus books whose
  verdict in ``chapter_book_verdicts.csv`` is ``exclude``. Books with no
  Chapter layer are not in the dataset (nothing to learn from).
* Books already in ``tsawa/data/processed/split_v3_frozen.csv`` keep their split.
* The remaining books are placed greedily, largest first, into the split with
  the largest relative deficit against 83/8.5/8.5 by window count
  (8192 tokens / 5120 stride), as for Sabche.
* The only constraint is that a book never sits in more than one split.
  Text shared between different books is not treated as leakage.

Usage:
    python chapter/src/prepare_chapter_split.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from build_tsawa_dataset import sliding_windows  # noqa: E402

AUDIT = ROOT / "tsawa/data/processed/tsawa_audit.csv"
CLEAN = ROOT / "chapter/data/processed/chapter_spans_clean.csv"
VERDICTS = ROOT / "chapter/data/processed/chapter_book_verdicts.csv"
SPLIT_V3 = ROOT / "tsawa/data/processed/split_v3_frozen.csv"
OUT_SPLIT = ROOT / "chapter/data/processed/chapter_split_frozen.csv"
OUT_REPORT = ROOT / "scratch/chapter/splits/chapter_split_report.json"
TOKENIZER = "jhu-clsp/mmBERT-base"
CONTENT_LEN = 8190  # 8192 - CLS - SEP
STRIDE = 5120
SEED = 123
TARGETS = {"train": 0.83, "val": 0.085, "test": 0.085}


def main() -> int:
    audit = {r["pecha_id"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8"))}
    verdict = {r["pecha_id"]: r for r in csv.DictReader(VERDICTS.open(encoding="utf-8"))}
    v3 = {r["pecha_id"]: r["split"] for r in csv.DictReader(
        l for l in SPLIT_V3.open(encoding="utf-8") if not l.startswith("#"))}
    excluded = {p for p, v in verdict.items() if v["verdict"] == "exclude"}

    spans = defaultdict(int)
    for r in csv.DictReader(CLEAN.open(encoding="utf-8")):
        if r["dropped"] != "True" and r["pecha_id"] not in excluded:
            spans[r["pecha_id"]] += 1
    books = sorted(spans)
    print(f"books kept {len(books)}  excluded {len(excluded)}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    tok.model_max_length = int(1e12)
    n_tok, n_win = {}, {}
    for p in books:
        t = Path(audit[p]["base_path"]).read_text(encoding="utf-8")
        n_tok[p] = len(tok(t, add_special_tokens=False)["input_ids"])
        n_win[p] = len(sliding_windows(n_tok[p], CONTENT_LEN, STRIDE))

    assign = {p: v3[p] for p in books if p in v3}
    free = [p for p in books if p not in v3]
    total_w = sum(n_win.values())
    cur = Counter()
    for p, sp in assign.items():
        cur[sp] += n_win[p]
    rng = random.Random(SEED)
    free.sort(key=lambda p: (-n_win[p], rng.random()))
    for p in free:
        sp = max(TARGETS, key=lambda s: (TARGETS[s] * total_w - cur[s]) / TARGETS[s])
        assign[p] = sp
        cur[sp] += n_win[p]

    stats = {}
    for sp in TARGETS:
        m = [p for p in books if assign[p] == sp]
        stats[sp] = {
            "books": len(m), "old": sum(p.startswith("P") for p in m),
            "new": sum(p.startswith("I") for p in m),
            "windows": sum(n_win[p] for p in m),
            "pct_windows": round(100 * sum(n_win[p] for p in m) / total_w, 1),
            "chapter_spans": sum(spans[p] for p in m),
            "from_split_v3": sum(p in v3 for p in m),
        }
        s = stats[sp]
        print(f"{sp:5} books={s['books']:3} (old {s['old']}, new {s['new']}, v3 {s['from_split_v3']}) "
              f"windows={s['windows']} ({s['pct_windows']}%) spans={s['chapter_spans']:,}")

    OUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SPLIT.open("w", newline="", encoding="utf-8") as fh:
        fh.write(
            "# chapter layer-detection document split — FROZEN\n"
            f"# generated: {date.today().isoformat()}\n"
            "# generator: chapter/src/prepare_chapter_split.py\n"
            f"# seed: {SEED}\n"
            "# base: split_v3_frozen.csv assignments kept\n"
            "# constraint: one book, one split. No grouping of books that share text.\n"
            f"# targets for new books: {TARGETS['train']:.0%}/{TARGETS['val']:.1%}/{TARGETS['test']:.1%} "
            "by WINDOW count (8192 / 5120)\n"
            "# span source: chapter/data/processed/chapter_spans_clean.csv (dropped=False)\n"
            "# TEST SPLIT IS FROZEN: evaluate test once at the end, never for tuning.\n")
        w = csv.DictWriter(fh, fieldnames=["pecha_id", "split", "n_tokens", "n_windows", "n_chapter_spans"])
        w.writeheader()
        for p in books:
            w.writerow({"pecha_id": p, "split": assign[p], "n_tokens": n_tok[p],
                        "n_windows": n_win[p], "n_chapter_spans": spans[p]})
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps({"stats": stats, "excluded": sorted(excluded)}, indent=2))
    print(f"wrote {OUT_SPLIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
