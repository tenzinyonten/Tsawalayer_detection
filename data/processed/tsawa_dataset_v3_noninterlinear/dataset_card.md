# tsawa binary BIO dataset — v3 (interlinear commentaries out of train)

- **Built:** 2026-09-18 by `src/build_tsawa_dataset.py`
- **Span source:** `data/processed/tsawa_spans_resolved.csv`
- **Split file:** `data/processed/split_v2_frozen.csv` (frozen v2 split, seed 123)
- **Tokenizer:** `jhu-clsp/mmBERT-base`, `max_length=8192`, `stride=5120`
- **Labels:** `O=0`, `B-TSAWA=1`, `I-TSAWA=2`; special tokens and padding `-100`
- **Excluded from train:** `I1637B774`, `I881A57E8`, `I9FF2B59B`, `IDAD44BA2`, `IF3ACC3E1` (5 documents, via `--exclude-pechas`)
- **Validation and test are unchanged from v2** — same documents, windows and
  labels, verified byte-for-byte on label counts. The frozen test split holds.

## Why these five were dropped

They are interlinear commentary (མཆན་འགྲེལ, *mchan-'grel*): the root text is
split into short chunks interleaved with gloss, so tsawa annotations land as
7–11 character fragments rather than verse lines. Median span length is 7–11
chars for these books versus 26+ for every other document, with nothing in
between — a genre boundary, not a noise threshold. Stitching `IF3ACC3E1`'s
spans in document order reconstructs a continuous root text, and its
`meta.yml` title contains མཆན་འགྲེལ; two others carry མཆན in the title.

`I895C519A` is the same genre but sits in **test**, so it was deliberately
kept: the frozen test split must not change across experiments. Test metrics
therefore still include one interlinear book, which is the honest comparison.

## Counts vs v2

| split | docs | windows (v2) | spans (v2) | positive % (v2) | median span chars (v2) |
|---|---:|---:|---:|---:|---:|
| train | 149 (154) | 5,331 (5,376) | 12,746 (17,430) | 4.623% (4.759%) | 116 (80) |
| validation | 27 (27) | 702 (702) | 2,046 (2,046) | 4.773% (4.773%) | 80 (80) |
| test | 31 (31) | 650 (650) | 1,673 (1,673) | 4.419% (4.419%) | 85 (85) |

## Label counts (padding excluded)

| split | O | B-TSAWA | I-TSAWA | real tokens |
|---|---:|---:|---:|---:|
| train | 41,619,635 | 20,147 | 1,997,236 | 43,637,018 |
| validation | 5,471,769 | 3,288 | 270,982 | 5,746,039 |
| test | 5,065,763 | 2,707 | 231,489 | 5,299,959 |

## Integrity

- 207 documents, 207 unique — no pecha appears in two splits.
- None of the five excluded IDs appears in any split.
- Validation and test match v2 exactly: True.

## Comparison builds

- `data/processed/tsawa_dataset/` — v1, coverage-quartile split (seed 42)
- `data/processed/tsawa_dataset_v2/` — frozen split, all 212 documents

