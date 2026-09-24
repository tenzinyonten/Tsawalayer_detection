---
pretty_name: Formatting Chapter
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- chapter
- bio
- mmbert
- openpecha
size_categories:
- 10K<n<100K
license: other
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Chapter (ལེའུ་, chapter heading) dataset — mmBERT BIO

Binary token classification for the Chapter layer, built with the same
labeling and windowing as the tsawa and sabche datasets. Generated 2026-09-24.

Split: 83/8.5/8.5 by window count (`chapter_split_frozen.csv`), by book only.

| item | value |
|---|---|
| tokenizer | `jhu-clsp/mmBERT-base` (fast, offsets) |
| window / stride | 8192 (8190 content + CLS/SEP) / 5120 |
| labels | `O`=0, `B-CHAPTER`=1, `I-CHAPTER`=2; CLS/SEP/pad = -100 |
| label rule | token-start rule (`build_tsawa_dataset.label_tokens`) |
| columns | input_ids, attention_mask, labels, token_start, token_end, char_start, char_end, pecha_id, source_batch, window_index, n_tokens_doc, coverage_pct |
| span source | `chapter/data/processed/chapter_spans_clean.csv` (dropped=False) |
| split | `chapter/data/processed/chapter_split_frozen.csv` (**test frozen**) |

## Size

| split | books | windows | Chapter spans |
|---|---|---|---|
| train | 304 (115 old, 189 new) | 9,188 | 2,292 |
| validation | 37 (15 old, 22 new) | 941 | 477 |
| test | 37 (13 old, 24 new) | 941 | 273 |

Train tokens: 75.0M `O`, 3,339 `B-CHAPTER`, 120,869 `I-CHAPTER` (windows overlap,
so a token is counted once per window it appears in). Chapter is about 0.16% of
tokens, far sparser than the other layers. Inverse-frequency weights from train
are about 7,500 (B) and 207 (I) against 0.33 (O).

## Pipeline

```bash
python chapter/src/clean_chapter_spans.py      # sidecar + book verdicts
python chapter/src/prepare_chapter_split.py    # frozen book-level split
python chapter/src/build_chapter_dataset.py
```

## Cleaning

Raw `Chapter.yml` offsets from 394 books with a Chapter layer (159 old, 235 new;
3,252 spans). Books with no Chapter layer are not in the dataset.

- **Edge snapping.** A start or end cut inside a syllable is walked to the nearest
  boundary within 3 characters (316 spans). Spans stop before the trailing shad.
  The Tsawa boundary checker reports about 72% dirty ends on the raw data; that
  is an artifact of the shad convention. Only about 8% of ends and 3.6% of starts
  (old batch) were actually mid-syllable.
- **Merging.** Only spans separated by shad, tsheg or spaces (never a newline) are
  merged: 5 merges. Each line is its own heading.
- **Book exclusions (16).** One book has unfixable edges. Fifteen old-batch books
  have offsets that drift partway through: the first few spans are correct and
  every span after some point is cut mid-sentence. A constant shift does not
  recover them, so the books are left out entirely. They hold 27 correct spans
  in the five largest cases checked (out of 131 spans).
- **Short spans are kept.** 306 raw spans are under 15 characters (short chapter
  numbers like `ལེའུ་བརྒྱད་པ`), plus a few 2-character stubs such as `༼ཇ`.
- **Splitting.** Books already in tsawa split v3 keep their split; the rest are placed
  by window count. Text shared between different books is not treated as leakage
  (about 10% of long Chapter spans also appear in another book); only a book never
  appears in more than one split.

## Known limitations

- Most books have very few Chapter spans (216 of 393 have one or two before
  exclusions), and some may be under-annotated. Not audited.
- About 1.1% of new-batch Chapter spans overlap a BookTitle span. Not resolved.
- The test split has only 273 spans in 37 books, so scores will be noisy.
- The heading-shape check used to find drifted books is a heuristic.
