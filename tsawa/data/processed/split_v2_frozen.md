# Document split v2 (frozen)

- **Generated:** 2026-09-15
- **Generator:** `scratch/tsawa/scripts/build_stratified_split.py`, greedy multi-objective, **seed 123**
- **Targets:** 80/10/10 by **window** count (not document count)
- **Stratified on:** batch (`P`=old / `I`=new), `n_windows`, tsawa char density, `pct_short`
- **Must-link groups honoured:** 11 reprint groups
- **Assignment file:** `tsawa/data/processed/split_v2_frozen.csv`
- **Span source:** `tsawa/data/processed/tsawa_spans_resolved.csv`

> **The test split is frozen.** Its documents must not change between
> experiments. Evaluate test **once at the end**. Do not use it for tuning,
> early stopping, or model selection — that is what `val` is for.

## Why v2 replaced v1

v1 (`build_tsawa_dataset.py` default, seed 42) stratified by Phase-2
`coverage_pct` quartiles at **document count**. It did not balance window mass,
batch ratio, or span-length mix. Consequences on v1:

- token positive density **4.70 / 4.17 / 5.53 %** — test 33% denser than val,
  so scores were not comparable across splits
- old-batch window share **60 / 25 / 24 %** — val and test were mostly new-batch
- `pct_short` **32 / 11 / 38 %** — the short-fragment cluster was wildly uneven
- 20-document eval sets, too thin to be stable

## Comparison (character density, from `scratch/pecha_profile.csv`)

### v1 — coverage-quartile document split (seed 42)

| split | docs (new/old) | windows | win% | char density | pct_short | old-batch win% |
|---|---|---:|---:|---:|---:|---:|
| train | 172 (95/77) | 5463 | 81.2% | 4.510% | 32.47% | 60.4% |
| val | 20 (14/6) | 615 | 9.1% | 3.975% | 10.91% | 24.7% |
| test | 20 (14/6) | 650 | 9.7% | 5.309% | 38.08% | 23.7% |

### v2 — frozen window-balanced split (seed 123)

| split | docs (new/old) | windows | win% | char density | pct_short | old-batch win% |
|---|---|---:|---:|---:|---:|---:|
| train | 154 (88/66) | 5376 | 79.9% | 4.570% | 31.62% | 53.6% |
| val | 27 (15/12) | 702 | 10.4% | 4.541% | 31.04% | 53.6% |
| test | 31 (20/11) | 650 | 9.7% | 4.273% | 29.71% | 53.2% |

Global: char density **4.538%**, `pct_short` **31.42%**, old-batch window share **53.6%**, 6728 windows over 212 pechas.

Val-vs-test char-density gap: **1.33 pp → 0.27 pp**.

Character density is `tsawa_chars / doc_chars`. Token-level positive density is
recorded in `tsawa/data/processed/tsawa_dataset_v2/dataset_card.md` after the rebuild.

## Must-link groups

Pechas sharing a normalized `source_metadata.title` are reprints/volumes of one
work and must stay in the same split (leakage). Groups with a **generic**
publisher-imprint title were deliberately **not** linked — e.g. 20 `P0000*`
volumes all titled `བློ་གྲོས་མཐའ་ཡས་པའི་མཛོད།` are separate works from one
imprint; gluing them collapsed val density to 1.07%.

- `test`: I2133CA39, IF428CDDB
- `train`: I36A7A668, I7D476A8B
- `train`: I40E5024E, P000094
- `test`: I4B1806B4, I8698A1DF
- `train`: I4FABF1F8, I8961718B
- `train`: I4FD99A33, I6380CEC4, I8B49E87B
- `train`: I52248444, IDBFF9DE3
- `val`: ICDC84458, IE5895799
- `train`: P000070, P000151
- `train`: P000073, P000074, P000101
- `val`: P000179, P000199

## Integrity

- 212 pechas, 212 unique — no pecha in two splits.
- No must-link group is split across splits.

