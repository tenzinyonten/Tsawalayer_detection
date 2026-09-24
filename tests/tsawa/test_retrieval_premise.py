#!/usr/bin/env python3
"""
test_retrieval_premise.py — can tsawa be found by matching instead of tagging?

The premise: a commentary quotes a root text, and that root text is often a
published work in its own right. DharmaCloud publishes them separately
(LEK-PHI-059-2 is the Beacon of Certainty; 059-1 is its commentary), so some
of them are probably already among the 539 cloned repos.

If a tsawa span appears verbatim in ANOTHER book, then root-text detection can
be framed as alignment against a corpus rather than classification from style.
Alignment cannot hallucinate a boundary, which is exactly the failure mode
eating precision right now.

This tests the premise only. It does not build the retrieval system; it asks
whether there is anything to retrieve.

Method: take tsawa spans from commentary books, normalise them, and look for
them as substrings of every other book. Report how many are found and where.

Usage
-----
    python test_retrieval_premise.py --sample 300
    python test_retrieval_premise.py --sample 300 --min-len 80 --partial
"""

from __future__ import annotations

import argparse
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def norm(s: str) -> str:
    """NFC, drop whitespace. Keeps tsheg and shad — they carry structure."""
    return "".join(unicodedata.normalize("NFC", s).split())


def load_all_texts(raw_opf: Path) -> dict[str, str]:
    texts = {}
    for d in sorted(raw_opf.iterdir()):
        if not d.is_dir() or not d.name.endswith(".opf"):
            continue
        pid = d.name[:-4]
        for cand in (d / f"{pid}.opf" / "base" / "v001.txt",
                     d / "base" / "v001.txt"):
            if cand.is_file():
                try:
                    texts[pid] = cand.read_text(encoding="utf-8")
                except Exception:
                    pass
                break
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="data/processed/tsawa/tsawa_spans_resolved.csv")
    ap.add_argument("--raw-opf", default="data/raw_opf")
    ap.add_argument("--sample", type=int, default=300,
                    help="tsawa spans to test")
    ap.add_argument("--min-len", type=int, default=60,
                    help="ignore spans shorter than this; short strings match "
                         "by chance and tell you nothing")
    ap.add_argument("--partial", action="store_true",
                    help="also try the first 40 chars, to catch near-matches "
                         "where the edition differs slightly")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="scratch/tsawa/analysis/retrieval_premise.csv")
    args = ap.parse_args()

    random.seed(args.seed)

    print("loading base texts...")
    texts = load_all_texts(Path(args.raw_opf))
    print(f"  {len(texts)} books")

    # normalised copies for searching, built once
    ntexts = {pid: norm(t) for pid, t in texts.items()}

    d = pd.read_csv(args.spans)
    if "dropped" in d.columns:
        d = d[~d.dropped.astype(bool)]
    d = d.copy()
    d["len"] = d.end - d.start
    pool = d[(d["len"] >= args.min_len) & (d.pecha_id.isin(texts))]
    print(f"  {len(pool):,} tsawa spans at least {args.min_len} chars")

    idx = random.sample(range(len(pool)), min(args.sample, len(pool)))
    sample = pool.iloc[idx]

    rows = []
    found_exact = found_partial = 0
    hit_books = Counter()
    hits_by_source = defaultdict(set)

    for n, (_, r) in enumerate(sample.iterrows(), 1):
        src = r.pecha_id
        span = norm(texts[src][int(r.start):int(r.end)])
        if len(span) < 20:
            continue

        matches = [pid for pid, t in ntexts.items()
                   if pid != src and span in t]
        partial = []
        if not matches and args.partial and len(span) > 40:
            head = span[:40]
            partial = [pid for pid, t in ntexts.items()
                       if pid != src and head in t]

        if matches:
            found_exact += 1
            for m in matches:
                hit_books[m] += 1
                hits_by_source[src].add(m)
        elif partial:
            found_partial += 1

        rows.append({
            "source_pecha": src, "start": int(r.start), "end": int(r.end),
            "len": int(r["len"]),
            "exact_matches": len(matches), "partial_matches": len(partial),
            "match_books": ",".join(matches[:5]),
            "span_head": texts[src][int(r.start):int(r.start) + 40],
        })

        if n % 50 == 0:
            print(f"  {n}/{len(sample)}  exact so far: {found_exact}", flush=True)

    total = len(rows)
    print(f"\n{'='*64}")
    print(f"RESULT — {total} tsawa spans tested")
    print("=" * 64)
    print(f"  found verbatim in another book: {found_exact} "
          f"({100*found_exact/max(total,1):.1f}%)")
    if args.partial:
        print(f"  first 40 chars found only:      {found_partial} "
              f"({100*found_partial/max(total,1):.1f}%)")
    print(f"  not found anywhere:             "
          f"{total - found_exact - found_partial} "
          f"({100*(total-found_exact-found_partial)/max(total,1):.1f}%)")

    if hit_books:
        print(f"\nbooks that contain other books' tsawa (candidate root texts):")
        for pid, c in hit_books.most_common(12):
            print(f"  {pid}: matched {c} spans")

    if hits_by_source:
        print(f"\ncommentary -> root pairs found: {len(hits_by_source)}")
        for src, dsts in list(hits_by_source.items())[:8]:
            print(f"  {src} -> {', '.join(sorted(dsts)[:3])}")

    print(f"\n{'='*64}")
    if found_exact / max(total, 1) > 0.25:
        print("PREMISE HOLDS for a useful fraction. Retrieval against a")
        print("root-text corpus is worth building: alignment cannot")
        print("hallucinate a boundary, which is the current failure mode.")
    elif found_exact / max(total, 1) > 0.05:
        print("PARTIAL. Some root texts are in the corpus, most are not.")
        print("Retrieval could be a precision-boosting auxiliary signal")
        print("rather than a standalone method. Widening the index to BDRC")
        print("or the Kangyur/Tengyur would test the ceiling.")
    else:
        print("PREMISE FAILS on this corpus. The root texts these")
        print("commentaries quote are mostly not among the 539 repos, so")
        print("there is nothing local to align against. Either widen the")
        print("index or drop the idea.")
    print("Note: exact substring matching is strict — different editions,")
    print("orthography or punctuation will miss. A fuzzy matcher would")
    print("find more, so treat this number as a lower bound.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()