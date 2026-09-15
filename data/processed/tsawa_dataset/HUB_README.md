---
pretty_name: Formatting Tsawa v1
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- tsawa
- bio
- mmbert
- openpecha
size_categories:
- 1K<n<10K
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

# Formatting Tsawa v1

Binary **BIO** token-classification data for Tibetan **tsawa** (རྩ་བ, root text) inside OpenPecha `.opf` books.

Tokenized with [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base). Splits are **by document** (pecha), not by window, so the same book never appears in two splits.

```python
from datasets import load_dataset

ds = load_dataset("Yontenn/formatting-tsawa-v1")
print(ds)
# DatasetDict({train, validation, test})
```

## Splits

| split | documents | windows | positive tokens | positive token % |
|-------|----------:|--------:|----------------:|-----------------:|
| train | 172 | 5,463 | 2,098,854 / 44,698,366 | 4.696% |
| validation | 20 | 615 | 209,672 / 5,031,053 | 4.168% |
| test | 20 | 650 | 294,202 / 5,322,147 | 5.528% |

Windows are length **8192** (including CLS/SEP/padding) with stride **5120**.

## Labels

| id | name |
|---:|------|
| 0 | `O` |
| 1 | `B-TSAWA` |
| 2 | `I-TSAWA` |
| -100 | special tokens and padding (ignored in loss) |

A token that straddles a span boundary is labeled from its **character start** offset only.

`isverse` is **not** used: the old batch has no `isverse` field, so using it would leak batch identity.

## Columns

- `input_ids`, `attention_mask`, `labels` — length 8192
- `pecha_id`, `source_batch` (`new` / `old`), `window_index`
- `token_start`, `token_end` — window slice in the full-document token sequence
- `char_start`, `char_end` — corresponding character offsets in `base/v001.txt`
- `n_tokens_doc`, `coverage_pct` — document-level audit fields

## Source and cleaning

- 212 OpenPecha repos with a Tsawa layer (123 new `I…` + 89 old `P000*`).
- Character spans snapped to Tibetan punctuation/whitespace boundaries, then overlapping spans resolved. Six overlap stubs dropped. Cloned `Tsawa.yml` files were not rewritten.
- Tokenizer: `jhu-clsp/mmBERT-base`. Document split seed: `42`, stratified by tsawa coverage quartiles.

High-density outlier `IFE0B60AA` (~96.7% tsawa coverage) is in **train** only (5 windows; ~1.9% of train positive tokens).

## Load locally (this repo)

Built by `src/build_tsawa_dataset.py` and saved with `save_to_disk` at `data/processed/tsawa_dataset/`.
