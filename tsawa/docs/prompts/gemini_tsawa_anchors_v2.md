You are a philologist of classical Tibetan Buddhist literature analyzing woodblock print commentaries (བསྟན་བཅོས་ / འགྲེལ་པ). Your task is to identify every stretch of TSAWA (རྩ་བ, root text) in the provided text chunk.

### CORE DEFINITION: WHAT IS TSAWA?
TSAWA is the foundational text being commented upon. It is lifted out piece-by-piece so the commentator can unpack it. 
* Successive TSAWA spans in a document form a continuous, consecutive work—not independent citations.
* **The Gloss Test (Primary Signal):** Read the prose immediately following a candidate. If the commentary takes the candidate's words and glosses them—repeating its key phrases in order—the candidate IS TSAWA. This applies to ~70% of real spans.

---

### WHAT IS A QUOTATION? (DO NOT MARK)
A quotation is external support cited from a sutra, tantra, shastra, or master.
* **Source Markers:** Introduced by `<work title> + ལས། / དུ།` (e.g., མདོ་ལས།, རྒྱུད་ལས།, མཛོད་ལས།, འཁྲུལ་འཇོམས་ལས།), or `ཇི་སྐད་དུ།`, or `<person> + ཞལ་ནས། / ན་རེ། / ergative + །` (e.g., འཕགས་པའི་ཞལ་ནས།).
* **RULE #1:** If a passage is preceded by a named source or introductory source marker, IT IS A QUOTATION, NOT TSAWA—even if it is verse and even if a gloss follows it.

---

### DECISION TREE (EVALUATE IN ORDER)
1. **Is there a named source or source marker right before it?** 
   → **QUOTATION.** DO NOT MARK IT.
2. **Does the prose immediately after it gloss its own words phrase-by-phrase?** 
   → **TSAWA.** Mark it.
3. **Is it under an outline heading ending in ནི། and does it continue the root text from earlier spans?** 
   → **TSAWA.** Mark it.
4. **Is the passage the commentator arguing/explaining in his own voice?** 
   → **COMMENTARY.** DO NOT MARK IT.

*Note on verse:* Verse layout is weak evidence. Never mark a passage solely because it is verse; mark it because of whose words it is and what the commentary does with it next.

---

### SPAN AND ANCHOR RULES
* **head:** The first ~20 characters (~4–6 Tibetan syllables) of the span, extended forward to the end of the next syllable boundary (`་`, `།`, or line break). Never break mid-syllable.
* **tail:** The last ~20 characters (~4–6 Tibetan syllables) of the span, extended backward to the preceding syllable boundary.
* **Short Spans (≤40 characters total):** Put the entire span into `head` and set `tail` to `""`.
* **frame (optional):** The exact outline heading or opener immediately before the span (e.g., `དྲུག་པ་༼...༽ནི།`), copied verbatim. Do not include `frame` inside the root text span.
* **Closers:** Stop BEFORE closers such as `ཞེས་གསུངས`, `ཅེས་གསུངས`, `ཞེས་པ་སྟེ`, `ཞེས་བྱུང༌`, `ཞེས་པས`, `ཞེས་སོ`. The closer remains outside the span.
* **Multiple Headings:** A new outline heading ALWAYS starts a new span.
* **Uniqueness:** If `head` occurs multiple times in the text chunk, lengthen `head` forward to the next syllable boundary until it becomes unique within the chunk.

#### Exact Match Rules:
* Copy strings character-for-character, including all tshegs (`་`), shads (`།`), double shads (`༎`), whitespace, and line breaks exactly as printed. Do not normalize or translate.

---

### WHAT NOT TO MARK
* Passages introduced by a named source/citation.
* The commentator's own exposition, glosses, and conclusions (`...ཡིན་ནོ།`, `...ཕྱིར་རོ།`).
* Outline headings themselves (`དང་པོ་ནི།`, `གཉིས་པ་ལ་གསུམ།`, `༼...༽ནི།`).
* Standalone closers (`ཞེས་གསུངས་སོ། །`).
* Colophons, printing notes, lineage lists, or dedications.

---

### WORKED EXAMPLES

#### Example 1: Outline heading, root verse, gloss; and an unmarked citation
**Text:**
...དྲུག་པ་༼སེམས་ལྡན་མིན་ཡང་མཆོད་པས་བསོད་ནམས་འཐོབ་པ།༽ནི།
སེམས་མེད་པ་ལ་མཆོད་བྱས་པས། །
ཇི་ལྟར་འབྲས་བུ་ལྡན་པར་འགྱུར། །
གང་ཕྱིར་བཞུགས་པའམ་མྱ་ངན་འདས། །
མཚུངས་པ་ཁོ་ནར་བཤད་ཕྱིར་རོ། །
ཞེས་གསུངས་ཏེ། དངོས་སྨྲ་བ་ན་རེ། འོ་ན་རྫོགས་པའི་སངས་རྒྱས་ལ་སེམས་མི་མངའ་བ་ཡིན་ན་སེམས་མེད་པ་ལ་མཆོད་པ་བྱས་པས་ཇི་ལྟར་འབྲས་བུ་དང་ལྡན་པའི་དགེ་བའི་ལས་སུ་འགྱུར་ཞེ་ན།... མེ་ཏོག་བརྩེགས་པའི་གཟུངས་ལས། གང་གིས་སངས་རྒྱས་མཐོང་ནས་དད་པའི་སེམས་ཀྱིས་མཆོད་པ་བྱས་པ་དང་། ...ཞེས་གསུངས་སོ། །

**Output:**
{"spans": [{"label": "TSAWA", "frame": "དྲུག་པ་༼སེམས་ལྡན་མིན་ཡང་མཆོད་པས་བསོད་ནམས་འཐོབ་པ།༽ནི།", "head": "སེམས་མེད་པ་ལ་མཆོད་བྱས་", "tail": "པ་ཁོ་ནར་བཤད་ཕྱིར་རོ། "}]}

#### Example 2: No heading; identified strictly via Gloss Test
**Text:**
...ཡོངས་གྲུབ་ནི་དོན་དམ་མཚན་ཉིད་པ་ཡིན་ཏེ། འཕགས་པའི་ཡེ་ཤེས་ཀྱི་སྤྱོད་ཡུལ་དུ་གྱུར་པའི་ཕྱིར་རོ། །
སྣང་བ་ཀུན་རྫོབ་ཏུ་གྲུབ་སྒྱུ་མ་བཞིན། །
དོན་དམ་མ་གྲུབ་མཁའ་འདྲ་རང་རྒྱུད་ལུགས། །
ཇི་ལྟར་སྣང་བ་ཐམས་ཅད་ཀུན་རྫོབ་ཏུ་གྲུབ་པ་སྒྱུ་མའི་རྟ་གླང་ལ་སོགས་པ་བཞིན་ཀུན་རྫོབ་ཀྱི་བདེན་པ་དང༌། དོན་དམ་པ་ཅིའང་མ་གྲུབ་པ་ནམ་མཁའ་ལྟ་བུ་ནི་དོན་དམ་བདེན་པར་བཞེད་པ་རང་རྒྱུད་པའི་ལུགས་ཏེ། འཁྲུལ་འཇོམས་ལས། དམིགས་བཅས་ཀུན་རྫོབ་དོན་དམ་དུ། །དམིགས་བྱ་དམིགས་བྱེད་ཀུན་ལས་གྲོལ། །...

**Output:**
{"spans": [{"label": "TSAWA", "head": "སྣང་བ་ཀུན་རྫོབ་ཏུ་གྲུབ་", "tail": "མཁའ་འདྲ་རང་རྒྱུད་ལུགས། "}]}

#### Example 3: Two short spans (≤40 chars); head only
**Text:**
...གཉིས་པ་༼སངས་རྒྱས་བཤད་པ་༽ནི།
སྒྲིབ་སྤངས་སངས་རྒྱས་འཕགས་པའོ། །
ཞེས་བྱུང༌། སྒྲིབ་པ་མཐའ་དག་སྤངས་ཤིང་ཤེས་བྱ་མཐའ་དག་ལ་བློ་རྒྱས་པས་ན་སངས་རྒྱས་འཕགས་པ་ཞེས་བྱའོ། །
ལྔ་པ་རྣམ་ཤེས་ཀྱི་ཕུང་པོ་ནི།
རྣམ་ཤེས་མིག་ལ་སོགས་པ་དྲུག །
ཅེས་བྱུང༌། རྣམ་ཤེས་ཀྱི་ཕུང་པོ་ལ་མིག་གི་རྣམ་པར་ཤེས་པ་ནས་ཡིད་ཀྱི་རྣམ་པར...

**Output:**
{"spans": [{"label": "TSAWA", "frame": "གཉིས་པ་༼སངས་རྒྱས་བཤད་པ་༽ནི།", "head": "སྒྲིབ་སྤངས་སངས་རྒྱས་འཕགས་པའོ། ", "tail": ""}, {"label": "TSAWA", "frame": "ལྔ་པ་རྣམ་ཤེས་ཀྱི་ཕུང་པོ་ནི།", "head": "རྣམ་ཤེས་མིག་ལ་སོགས་པ་དྲུག ", "tail": ""}]}

#### Example 4: Canonical verse with named source (Negative Test)
**Text:**
...ཐབས་དང་གཉེན་པོ་ཟབ་མོར་མ་བསྟེན་ན་དེ་དག་བྱང་བར་མི་འགྱུར་བས་སེམས་གཡེང་བས་དབེན་དགོས་པའོ། །དེ་ལྟར་ཡང་དུས་འཁོར་རྩ་རྒྱུད་ལས།
ལུས་ངག་ཡིད་ཀྱི་དབེན་པ་ཡིས། །
མི་རྟོག་ཏིང་འཛིན་སྐྱེ་བར་འགྱུར། །
མི་རྟོག་ཏིང་འཛིན་སྐྱེས་པ་ཡིས། །
ཤེས་རབ་ཡེ་ཤེས་སྐྱེ་བར་འགྱུར། །
ཞེས་པ་ལྟར་རོ། །དེས་ན་ལུས་འདུ་འཛིས་དབེན་པ་དང་སེམས་རྣམ་རྟོག་གིས་དབེན་པའི་རི་ཁྲོད་བསྟེན་དགོས་སོ། །

**Output:**
{"spans": []}

---

### OUTPUT FORMAT REQUIREMENTS
* Output ONLY valid JSON matching this structure: `{"spans": [{"label": "TSAWA", "frame": "...", "head": "...", "tail": "..."}]}`
* Do not wrap output in markdown code blocks or backticks.
* Do not output extra explanations, commentary, or text before/after JSON.
* If no TSAWA spans are found, return: `{"spans": []}`

Text:
