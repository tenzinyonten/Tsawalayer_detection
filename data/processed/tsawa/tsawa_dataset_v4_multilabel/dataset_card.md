# tsawa + quotation BIO dataset — v4 (5 labels)

- **Built:** 2026-09-18 by `src/tsawa/build_tsawa_dataset.py --quote-sidecar`
- **Tsawa spans:** `data/processed/tsawa/tsawa_spans_resolved.csv` (21,149 active)
- **Quotation spans:** `data/processed/tsawa/quotation_spans_snapped.csv` (29,538 active)
- **Split file:** `data/processed/tsawa/split_v2_frozen.csv` — document assignment unchanged from v2
- **Tokenizer:** `jhu-clsp/mmBERT-base`, `max_length=8192`, `stride=5120`
- **Labels:** `O=0`, `B-TSAWA=1`, `I-TSAWA=2`, `B-QUOTE=3`, `I-QUOTE=4`; special tokens and padding `-100`

## Why quotation was added

The binary model's largest error class was predicting spans that overlap no
gold tsawa (53.7% of predictions). Quotations share the framing particles and
verse structure of root text but had no label, so the model could not learn to
reject them. On 60 books carrying both layers the classes separate on surface
form: median length 32 vs 73 chars, isometric clauses 73.0% vs 89.5%, followed
by a closer 34.9% vs 57.5%, preceded by an attribution 8.0% vs 34.6%.

## Layer merge

The old batch names the layer `Quotation.yml` and the new batch `Citation.yml`.
They are treated as one class. Among the 212 tsawa books: 82 have
`Quotation.yml` (all old batch), 110 have `Citation.yml` (all new batch), and
`I8994AAB2` ships both with disjoint offsets, so the union is taken — **191
books with a quotation layer, 21 without**. Books without one keep their tsawa
annotation and contribute no QUOTE spans.

## Boundary snapping

The boundary bug is not Tsawa-specific. Raw quotation offsets were **32.8%**
dirty overall, and **64.6%** in the old batch (34.2% dirty start, 53.9% dirty
end) against 3.9% in the new batch. They were snapped with the identical rules
and `max_walk=12` used for tsawa, giving **0.0% dirty** on both edges. 4,804
starts and 8,183 ends moved. 184 same-class quote-quote overlaps were trimmed
and 18 fully-contained stubs dropped. Cloned YAML was not modified.

## Overlap and the precedence rule

**TSAWA wins every conflict**, since it is the target class. Overlapping
characters are labeled TSAWA and the quotation span is carved around them; a
quotation split in two by a tsawa span yields two runs, the second opening a
fresh `B-QUOTE`.

| | spans |
|---|---:|
| quotation spans in | 29,538 |
| untouched by any tsawa span | 26,565 |
| clipped (274 of them split in two) | 299 |
| **fully swallowed, dropped** | **2,674** |
| quotation spans out | 27,795 |

**119,513 characters / 84,354 tokens were suppressed** by the rule.

The swallowed spans are overwhelmingly new-batch (2,673 new vs 1 old): in the
new batch a quotation is frequently annotated *inside* a tsawa span, i.e. a
citation nested within root text, rather than as an alternative to it. Those
2,674 spans contribute no QUOTE tokens at all under this rule. If the quotation
class underperforms, that is the first thing to revisit.

## Windows and documents

| split | documents | windows | books with a quotation layer |
|---|---:|---:|---:|
| train | 154 | 5,376 | 139 |
| validation | 27 | 702 | 25 |
| test | 31 | 650 | 27 |

Window counts are identical to v2 — the split and the windowing did not change.

## Span counts (after precedence)

| split | TSAWA spans | QUOTE spans | median TSAWA chars | median QUOTE chars |
|---|---:|---:|---:|---:|
| train | 17,430 | 23,150 | 80 | 110 |
| validation | 2,046 | 3,120 | 80 | 115 |
| test | 1,673 | 1,525 | 85 | 69 |

## Token counts by class (padding excluded)

| split | O | B-TSAWA | I-TSAWA | B-QUOTE | I-QUOTE | real tokens |
|---|---:|---:|---:|---:|---:|---:|
| train | 38,841,591 | 27,380 | 2,066,882 | 36,832 | 3,032,883 | 44,005,568 |
| validation | 5,013,933 | 3,288 | 270,982 | 4,931 | 452,905 | 5,746,039 |
| test | 4,895,414 | 2,707 | 231,489 | 2,374 | 167,975 | 5,299,959 |

## Token positive density by class

| split | TSAWA % | TSAWA % in v2 | QUOTE % | any positive % |
|---|---:|---:|---:|---:|
| train | 4.759% | 4.759% | 6.976% | 11.735% |
| validation | 4.773% | 4.773% | 7.968% | 12.741% |
| test | 4.419% | 4.419% | 3.214% | 7.633% |

The TSAWA columns are identical to v2: adding the quotation class did not move
a single tsawa label. QUOTE density is far less even across splits than TSAWA
is — the frozen split was stratified on tsawa density, not quotation density.

## Integrity

- 212 documents, 212 unique — no pecha appears in two splits.
- TSAWA token counts match v2 exactly: True.
- Merged spans are verified non-overlapping per document at build time.

## Comparison builds

- `data/processed/tsawa/tsawa_dataset_v2/` — 3-label, all 212 documents, same split
- `data/processed/tsawa/tsawa_dataset_v3_noninterlinear/` — 3-label, 5 interlinear books out of train

