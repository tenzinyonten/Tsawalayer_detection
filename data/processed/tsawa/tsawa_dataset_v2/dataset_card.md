# tsawa binary BIO dataset — v2 (frozen split)

- **Built:** 2026-09-15 by `src/tsawa/build_tsawa_dataset.py`
- **Span source:** `data/processed/tsawa/tsawa_spans_resolved.csv` (snapped boundaries + overlap resolve; `dropped=True` skipped)
- **Split file:** `data/processed/tsawa/split_v2_frozen.csv` (greedy window-balanced, seed 123)
- **Repo selection:** `data/processed/tsawa/tsawa_audit.csv`, `--source combined` (212 pechas: 123 new + 89 old)
- **Tokenizer:** `jhu-clsp/mmBERT-base`, `max_length=8192`, `stride=5120`
- **Labels:** `O=0`, `B-TSAWA=1`, `I-TSAWA=2`; special tokens and padding `-100`
- **Straddle rule:** token labeled by its **character start** offset
- **Excluded spans:** 6 overlap stubs → `data/processed/tsawa/dropped_spans_v2.csv`; 21,149 spans labeled
- Cloned `Tsawa.yml` files were not modified. No model was trained.

> **The test split is frozen.** Do not change its documents between
> experiments and do not use it for tuning or model selection.

## Splits

| split | documents | windows | real tokens | positive (B+I) | **positive %** | v1 positive % |
|---|---:|---:|---:|---:|---:|---:|
| train | 154 | 5,376 | 44,005,568 | 2,094,262 | **4.759%** | 4.696% |
| validation | 27 | 702 | 5,746,039 | 274,270 | **4.773%** | 4.168% |
| test | 31 | 650 | 5,299,959 | 234,196 | **4.419%** | 5.528% |

Token positive-density spread across splits: **v1 1.36 pp → v2 0.35 pp**.

## Label counts (padding excluded)

| split | O | B-TSAWA | I-TSAWA | `-100` slots | padding % of slots |
|---|---:|---:|---:|---:|---:|
| train | 41,911,306 | 27,380 | 2,066,882 | 34,624 | 0.054% |
| validation | 5,471,769 | 3,288 | 270,982 | 4,745 | 0.058% |
| test | 5,065,763 | 2,707 | 231,489 | 24,841 | 0.442% |

## Short-span mix (spans < 30 chars, measured on this split)

| split | spans | short | pct_short |
|---|---:|---:|---:|
| train | 17,430 | 5,512 | 31.62% |
| validation | 2,046 | 635 | 31.04% |
| test | 1,673 | 497 | 29.71% |

## Integrity

- Documents: 212 across splits, 212 unique — no pecha appears in two splits.
- Window counts match the profile prediction (5376 / 702 / 650): True.
- Must-link reprint groups are kept within one split (see `data/processed/tsawa/split_v2_frozen.md`).

## Relationship to v1

`data/processed/tsawa/tsawa_dataset/` (v1) is kept for comparison. Same spans and
tokenizer; it differs only in the document split (coverage-quartile, seed 42).

## `isverse`

Not used: the old batch has no `isverse` field, so it would encode batch identity.

