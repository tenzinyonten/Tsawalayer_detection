"""Scan every tsawa book for verse-metre runs that are NOT labelled tsawa (possible missed tsawa).
Read-only on data/. Writes missed_tsawa_by_book.csv and missed_tsawa_summary.md next to this file.

A run = 4+ consecutive shad-delimited clauses, all with a 7/9/11/13 syllable metre, none touching a tsawa span.
  loose  : any mix of those four metres
  strict : every clause in the run has the SAME syllable count (isometric stanza), much rarer in prose
Runs overlapping a Quotation or Citation span are reported separately (verse quoted from another work is
legitimately not tsawa)."""
import glob, os, re
import pandas as pd, yaml

R = "data/raw_opf"; OUT = "common/annotation_audit"
SH = "།༎༏༐༑༔"; ISO = {7, 9, 11, 13}; MINRUN = 4
SHAD_RE = re.compile(f"[{SH}]")
def syl(s): return len([x for x in s.split("་") if x.strip()])

def spans(d, name, with_verse=False):
    p = f"{d}/layers/v001/{name}.yml"
    if not os.path.exists(p): return []
    y = (yaml.safe_load(open(p, encoding="utf-8")) or {}).get("annotations") or {}
    return sorted((a["span"]["start"], a["span"]["end"], a.get("isverse")) if with_verse else (a["span"]["start"], a["span"]["end"]) for a in y.values())

def overlaps(s, e, S): return any(a < e and s < b for a, b, *_ in S)

def runs(b, mask, strict):
    out, cur, p = [], [], 0
    def flush():
        if len(cur) >= MINRUN: out.append((cur[0][0], cur[-1][1], cur[0][2]))
    for m in SHAD_RE.finditer(b):
        n = syl(b[p:m.start()]); cs, ce = p, m.start(); p = m.end()
        if not n: continue
        ok = n in ISO and not any(mask[cs:ce])
        if ok and cur and strict and n != cur[-1][2]:
            flush(); cur.clear()
        if ok: cur.append((cs, ce, n))
        else: flush(); cur.clear()
    flush()
    return out

audit = pd.read_csv("tsawa/data/processed/tsawa_audit.csv")
split = pd.read_csv("tsawa/data/processed/split_v3_frozen.csv", comment="#").set_index("pecha_id").split.to_dict()
rows = []
for pid in audit[audit.has_tsawa_layer == True].pecha_id:
    d = [x for x in glob.glob(f"{R}/{pid}.opf/*") if x.endswith(".opf")][0]
    b = open(d + "/base/v001.txt", encoding="utf-8").read()
    T = spans(d, "Tsawa"); Qt = spans(d, "Quotation") + spans(d, "Citation"); C = spans(d, "Commentary", True)
    mask = bytearray(len(b))
    for s, e in T: mask[s:e] = b"\x01" * (e - s)
    tl = sum(e - s for s, e in T)
    rec = dict(pecha_id=pid, split=split.get(pid, "dropped"), text_chars=len(b), n_tsawa=len(T), tsawa_chars=tl,
               has_commentary=bool(C))
    for name, strict in (("strict", True), ("loose", False)):
        R_ = runs(b, mask, strict)
        quoted = [r for r in R_ if overlaps(r[0], r[1], Qt)]
        cverse = [r for r in R_ if any(a < r[1] and r[0] < c and v for a, c, v in C)]
        chars = sum(e - s for s, e, _ in R_)
        rec.update({f"{name}_runs": len(R_), f"{name}_chars": chars,
                    f"{name}_runs_in_quotation": len(quoted), f"{name}_runs_in_commentary_verse": len(cverse),
                    f"{name}_runs_unexplained": len([r for r in R_ if r not in quoted])})
    rows.append(rec)
df = pd.DataFrame(rows)
df["strict_chars_per_tsawa_char"] = (df.strict_chars / df.tsawa_chars.where(df.tsawa_chars > 0)).round(2)
df["strict_runs_per_100k"] = (df.strict_runs / df.text_chars * 1e5).round(1)
df.sort_values("strict_chars", ascending=False).to_csv(f"{OUT}/missed_tsawa_by_book.csv", index=False)
print(len(df), "books;", "strict runs total", int(df.strict_runs.sum()), "loose", int(df.loose_runs.sum()))

# ---------------- summary markdown ----------------
def md(t, cols, names):
    h = "| " + " | ".join(names) + " |\n|" + "|".join("---" for _ in names) + "|\n"
    return h + "".join("| " + " | ".join(f"{r[c]:,}" if isinstance(r[c], (int,)) and not isinstance(r[c], bool) else str(r[c]) for c in cols) + " |\n" for _, r in t.iterrows())
df["ratio"] = (df.strict_chars / df.tsawa_chars.where(df.tsawa_chars > 0)).round(1)
cols = ["pecha_id", "split", "n_tsawa", "tsawa_chars", "strict_runs_unexplained", "strict_runs_in_quotation", "strict_chars", "ratio", "text_chars"]
names = ["book", "split", "tsawa spans", "tsawa chars", "unlabelled verse runs", "runs inside Quotation/Citation (not counted)", "chars in all runs", "run chars per tsawa char", "book chars"]
dropped = df[df.split == "dropped"]
A = dropped[dropped.strict_runs_unexplained >= 20].sort_values("strict_runs_unexplained", ascending=False)
used = df[df.split != "dropped"]
B = used[(used.strict_runs_unexplained >= 50) & (used.ratio >= 5)].sort_values("strict_runs_unexplained", ascending=False)
Bev = B[B.split.isin(["val", "test"])]
ok = used[(used.strict_runs_unexplained >= 50) & (used.ratio < 2)]
L = ["# Where tsawa may be missing: unlabelled verse by book\n",
     f"Scan of all {len(df)} books with a Tsawa layer (`missed_verse_scan.py`; per-book numbers in `missed_tsawa_by_book.csv`). A **run** is 4 or more consecutive shad-delimited clauses that all have the same 7, 9, 11 or 13 syllable count (an isometric stanza) and do not touch any tsawa span. Runs that overlap a Quotation or Citation span are counted separately and left out of the main numbers, because verse quoted from another work is not tsawa.\n",
     "## What this does and does not show\n",
     f"- Totals: {int(df.strict_runs.sum()):,} runs in {len(df)} books (median {int(df.strict_runs.median())} per book; {int((df.strict_runs==0).sum())} books have none). By split: " + ", ".join(f"{k} {int(v):,}" for k, v in df.groupby('split').strict_runs.sum().items()) + ".",
     "- I read 9 random runs from 3 of the top books (`P000218`, `P000275`, `I2FCD4B1D`): all 9 are real verse, not noise. Several are introduced by a source marker (`བཤེས་སྤྲིངས་སུ།` 'in the Letter to a Friend', `སྡོམ་འབྱུང་ལས།` 'from the Sdom 'byung'), so they are quoted verse.",
     "- So a run is verse that nobody labelled as tsawa. It may be (a) missed tsawa, (b) a quotation from another text that was not marked Quotation, or (c) the author's own verse. The annotator decides which. Use the counts to choose where to look first, not as a count of errors.",
     "- To read a book's runs in context: `python common/annotation_audit/make_book_check.py <book id>` writes an HTML page with the tsawa spans highlighted and every run underlined in blue (the page uses the looser definition, any mix of the four metres, so it shows slightly more runs).\n",
     f"## 1. Dropped books: add tsawa from scratch ({len(A)} books with at least 20 unlabelled runs)\n",
     "These books have fewer than 10 tsawa spans, so they were left out of training. Most have hundreds of verse stanzas and almost no tsawa, so they are the clearest case for adding annotations.\n",
     md(A.head(30), cols, names),
     f"\n## 2. Training, validation and test books where tsawa is a small share of the verse ({len(B)} books)\n",
     "Criteria: at least 50 unlabelled runs and run characters at least 5 times the tsawa characters. Verse left unlabelled here teaches the model that verse is not tsawa (the `audit_before_v6` under-labelling concern). **The validation and test books matter most, since they also lower the measured score:** " + (", ".join(f"`{r.pecha_id}` ({r.split})" for r in Bev.itertuples()) or "none") + ".\n",
     md(B.head(30), cols, names),
     f"\n## 3. Probably fine\n",
     f"{len(ok)} training/validation/test books have at least 50 unlabelled runs but their tsawa already covers more than half as much text as the unlabelled verse (ratio under 2), for example `P000078` (ratio 0.9). Their unlabelled verse is likely quotation or the author's own stanzas next to a well-annotated root text. Listed in the CSV (`ratio` column).\n"]
open(f"{OUT}/missed_tsawa_summary.md", "w", encoding="utf-8").write("\n".join(L))
print("summary:", len(A), "dropped,", len(B), "used (", len(Bev), "val/test ),", len(ok), "fine")
