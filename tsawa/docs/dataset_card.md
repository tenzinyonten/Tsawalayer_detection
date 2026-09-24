# tsawa binary BIO dataset

Built by `common/build_tsawa_dataset.py` from Phase 2 `tsawa_audit.csv`.
**No model was trained.** Data-rights clearance is still an open question.

## Source

- `--source combined`: **123 new** + **89 old** repos (documents with a valid Tsawa layer per the audit).
- Tokenizer: `jhu-clsp/mmBERT-base`
- Windows: `max_length=8192`, `stride=5120` (token-index based; special tokens wrap each window).
- Document split seed: `42`
- Document-level 80/10/10 split, stratified by Phase 2 coverage_pct quartiles so val/test are not accidentally all low- or high-density (including outlier IFE0B60AA).

## Cleaning

- Labels from `/Users/tenzinyonten/layer_detection/tsawa/data/processed/tsawa_spans_resolved.csv`.
- Skip `dropped=True` (and empty stubs). **6** sidecar rows excluded; **21149** spans labeled.
- Cloned `Tsawa.yml` is not rewritten.
- Log of excluded sidecar rows: `tsawa/data/processed/dropped_spans.csv`.

## Labels

- `{'O': 0, 'B-TSAWA': 1, 'I-TSAWA': 2}`
- Special tokens and padding use label `-100`.
- Token / span straddles: labeled by the token **start** offset.

## `isverse`

Not used. The new batch has a complete true/false `isverse` field; the old batch is entirely missing it (6,845 spans). Using it would encode batch identity, not verse structure.

## High-density outlier `IFE0B60AA`

Phase 2 coverage ≈ 96.7% tsawa. It is windowed with the same max_length/stride as every other document (no special case). It can still contribute a large share of positive tokens/windows:

- `train`: 5 windows from `IFE0B60AA`; 1.9% of that split's positive (non-O) tokens.
- `validation`: 0 windows from `IFE0B60AA`; 0.0% of that split's positive (non-O) tokens.
- `test`: 0 windows from `IFE0B60AA`; 0.0% of that split's positive (non-O) tokens.

## Split composition

| split | documents | windows | positive tokens | positive token % |
|-------|----------:|--------:|----------------:|-----------------:|
| train | 172 | 5463 | 2,098,854 / 44,698,366 | 4.696% |
| validation | 20 | 615 | 209,672 / 5,031,053 | 4.168% |
| test | 20 | 650 | 294,202 / 5,322,147 | 5.528% |

Positive-token % is the class imbalance that matters for training; it is **not** the same as the document-level tsawa coverage % in the audit (~2.1% of all base text, ~4.5% among repos that have a Tsawa layer).

### Documents per split

- **train** (172): I0156A8B1, I069801F1, I07240379, I0B81CD66, I100E7DAD, I163178A0, I1637B774, I19A08A51, I1DCBD5CC, I2133CA39, I23323023, I25463E37, I27E347E9, I2D96E5C8, I2EAAF38A, I319DAFF7, I344D2A4C, I36A7A668, I40E5024E, I454D2699, I4B1806B4, I4FD99A33, I4FDF07E7, I5048861E, I51B9FE5F, I52248444, I575514A8, I587B7DED, I5BFCBB3F, I5F5D9F5A, I60D0ED54, I617FDD1E, I6D11E414, I6EFF9668, I74772DFF, I786395B4, I7D476A8B, I8062BDF5, I85485484, I85C28EDA, I8698A1DF, I881A57E8, I88977D82, I88CF073C, I8961718B, I8A57AE06, I8B49E87B, I91949F99, I920AA7AC, I9779A606, I97B90DC0, I9811592F, I98D03292, I9AEEF96A, I9B6A4525, I9CB71958, I9D9C7AC9, I9FF2B59B, IA2F1ACFA, IA3DD1ADA, IA473D353, IAA85B9B5, IAE1E11E5, IB45DE768, IB522F095, IB5F523B2, IB8D599AA, IB8EEB9FB, IBBB26EA1, IBC775AF6, IC05A6BE0, IC3A7006F, IC5CE7FE9, IC6F06BCD, IC87920B7, IC99C24DC, ICC9840B6, ID4AD2D0E, IDACD7FC0, IDAD44BA2, IDDA7F69E, IDDB46142, IE2421BEA, IE5895799, IE79CC998, IEE67309C, IEF3A4022, IF428CDDB, IF4C4BF01, IF6B54962, IF84B2E71, IF85C9484, IFA88A536, IFE0B60AA, IFF541F44, P000020, P000022, P000026, P000027, P000028, P000044, P000046, P000054, P000056, P000067, P000068, P000070, P000071, P000073, P000074, P000078, P000081, P000083, P000084, P000085, P000089, P000094, P000098, P000101, P000111, P000114, P000115, P000117, P000118, P000123, P000128, P000129, P000132, P000135, P000144, P000145, P000151, P000153, P000161, P000162, P000164, P000170, P000172, P000175, P000176, P000177, P000178, P000179, P000180, P000188, P000189, P000193, P000194, P000199, P000200, P000201, P000203, P000204, P000207, P000213, P000215, P000217, P000218, P000219, P000224, P000225, P000226, P000230, P000237, P000241, P000246, P000251, P000253, P000254, P000259, P000269, P000275

- **validation** (20): I058DD999, I3F4A91F5, I62D7430C, I849C345E, I895C519A, IB047F8C7, IB47565DE, IB8F4E5BE, IBF8C5BB6, IBFACCCDF, ICDC84458, IE166DA22, IEEBAD11E, IF7A04537, P000013, P000021, P000055, P000155, P000168, P000185

- **test** (20): I0FCFA88F, I2FCD4B1D, I4CAE7B8B, I4FABF1F8, I5134B437, I6380CEC4, I6F43CD1F, I88DA3111, I8994AAB2, I9779E60E, IC555D0EB, IDBFF9DE3, IEA9B747E, IF3ACC3E1, P000017, P000066, P000127, P000141, P000195, P000242
