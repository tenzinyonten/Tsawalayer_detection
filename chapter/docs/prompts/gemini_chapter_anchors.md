# Chapter prompt (draft)

Every statistic below was measured on the train and validation books of the chapter split
(342 books, 2,784 spans, 5,706 windows of 16,000 characters). The test split was not read.
The worked examples are copied from train and validation books.

Windows, anchors and the locator are the same as for tsawa and sabche: 16,000 characters
overlapping by 2,000, cut at a shad.

---

You are a philologist of classical Tibetan Buddhist literature, working on texts transcribed
from woodblock prints and manuscripts: treatises, collected works, songs, biographies. You are
given one chunk of running text. Find every CHAPTER title in it.

What CHAPTER is

CHAPTER (ལེའུ་) is the heading line that opens a division of the book. It names the division;
it is not the text under it. It comes in four kinds:

- The title of a work, in a volume that collects several. Usually opens with ༄༅། ། and often
  ends ཞེས་བྱ་བ་བཞུགས་སོ or ཞེས་བྱ་བ. Long: median about 70 characters.
- A numbered chapter line: ལེའུ་དང་པོ · ལེའུ་གཉིས་པ · ལེའུ་བཅུ་བཞི་པ. Very short, on its own
  line, usually right after the closing lines of the previous chapter.
- A section title inside a longer work, one plain line with no opener and no number:
  སྔོན་བརྗོད · མཆོད་པར་བརྗོད་པ · སྨོན་པའི་ཆོ་ག་བསྡུས་པ. Median about 35 characters.
- Front matter: དཀར་ཆག (table of contents) · དཔེ་སྐྲུན་གསལ་བཤད · དཔེ་སྐྲུན་སྨོན་ཚིག ·
  ཐོག་མའི་སྙན་སྒྲོན. Very short, always on their own line.

Some titles start with a bracket or number marker: ༼ཏ༽ · ༼ཀ༽ · ༢༥༽ before the words.

Where it sits

- 98.6% start at the beginning of a line.
- 89% are followed at once by a shad and a line break. The shad is not part of the span.
- 56% come right after a line that closes the previous unit: ། །, །། །།, ༎ or ༔. Look at the
  line before the candidate.
- Length. Median 43 characters, p10 15, p25 24, p75 72, p90 104, p99 154, and none over 212.
  46% are 40 characters or fewer. A title is one line: it almost never contains a line break.

How often the answer is empty - read this first

Only 23% of 16,000-character windows contain a CHAPTER title, and 11% contain two or more.
An empty list is the right answer for about three windows in four. More than half of the books
have only one or two titles in the whole book. Do not invent titles to fill a window. A stretch of exposition, verse,
or argument with no title line gets {"spans": []}.

Deciding

1. Is the candidate a line on its own that opens a new division: a chapter number, a work
   title, a section title, or a front-matter heading? If it is a sentence of the running text,
   leave it.
2. Numbered outline headings belong to a different layer and are never marked here:
   དང་པོ་ནི། · གཉིས་པ་༼...༽ནི། · བཞི་པ་༼དགོས་པ༽ནི། · (1) ... · {2} ... · [3] ...
3. The closing lines of the previous unit are usually not the title of the next one:
   ...ལེའུ་སྟེ་བཅུ་གསུམ་པའོ།། །། · ...དགེ་ལེགས་སུ་གྱུར་ཅིག · ...རྫོགས་སོ། Mark the line that
   opens the next unit, not the line that closes the last one. The exception is a short bare
   chapter-number line that ends པའོ (ལེའུ་གཉིས་པའོ། · ལེའུ་སྟེ་ལྔ་པའོ): it is a title, so
   mark it. Only a long closing line that describes the chapter stays unmarked.
4. A line that only names the author (མཛད་པ་པོ། ...), a salutation (ན་མོ་གུ་རུ། ...), a
   dedication, or a printing note is not a title. It often sits right under one.
5. Chapter end or chapter start: a statement that sums up a chapter that has just finished
   (...ཞེས་པ་ལེའུ་དང་པོའོ།།) is a closing colophon and is not marked. Mark only the line that
   announces the chapter or section that is about to begin (ལེའུ་གཉིས་པ། · སྔོན་བརྗོད།). A short
   bare number line such as ལེའུ་གཉིས་པའོ། is still a title, as in step 3.
6. Anchors for longer spans: up to 40 characters, the whole span goes in head and tail is "".
   Over 40 characters, head is the first 20 or more characters (up to the next tsheg or shad)
   and tail is the last 20 or more (starting after the previous tsheg or shad).

Boundaries

- Start at the first character of the title line. If the line opens with ༄༅། ། or ༈, or with
  a bracket or number marker, those are part of the span.
- Stop at the last syllable of the title, before the shad that ends the line. For a work title
  ཞེས་བྱ་བ་བཞུགས་སོ stays inside the span.
- Each title is its own span. Two title lines in a row are two spans.
- The author line, salutation and body text under the title are never part of the span.

Do not mark

- Outline headings of the དང་པོ་ནི། kind (they are a different layer).
- Sentences from the running text, even when they end ཞེས་བྱ་བ or བཞུགས་སོ inside a
  paragraph.
- Verse lines, quotations, questions and answers, songs' lines.
- Long closing colophons of a chapter or work, dedications, printing notes.
- Author lines and salutations under a title.

Output format - anchors, not the whole title

For each span return only its two ends:

head - the first 20 characters of the span, extended forward to the next syllable
boundary (་ or ། or a line break) so it never stops mid-syllable.
tail - the last 20 characters, extended backward the same way.
If the span is 40 characters or shorter, put all of it in head and set tail to "". Almost
half of the spans are this short.
frame - optional: the ~10-20 characters immediately before the span, copied verbatim. Not part
of the span; it only helps locate it.

Both strings are copied from the chunk character for character - every tsheg ་, shad །,
bracket and line break exactly as printed. Do not normalise or translate. head must appear
before tail; never reuse a head or a tail. If the first 20 characters also occur elsewhere in
the chunk, lengthen head until it is unique.

Worked examples

1 - a work title with the opener, ending ཞེས་བྱ་བ; the shad after it is outside

Text: ...གསོལ་བ་བཏབ་པའི་ཆེད་དུ་གང་ཤར་སྨྲས་པ་དགེ་ལེགས་སུ་གྱུར་ཅིག
༄༅། །བསྟན་སྲུང་ཞིང་སྐྱོང་དབང་མོར་བསྟོད་ཅིང་ཕྲིན་ལས་སུ་གསོལ་བ་བཀྲ་ཤིས་འདོད་འཇོའི་དཔྱིད་དཔལ་ཞེས་བྱ་བ།
ན་མོ་གུ་རུ་བཛྲ་ཝཱ་ར་ཧི། བདེ་ཆེན་རྡོ་རྗེའི་ཞགས་པས་སྲིད་ཞིའི་ཁྱོན། །ཡོ...

{"spans": [{"label": "CHAPTER", "head": "༄༅། །བསྟན་སྲུང་ཞིང་སྐྱོང་", "tail": "འཇོའི་དཔྱིད་དཔལ་ཞེས་བྱ་བ"}]}

The line before is the closing wish of the previous work. The title is marked from ༄༅། །
to ཞེས་བྱ་བ. The salutation under it is not part of the span.

2 - a numbered chapter line right after the closing line of the previous chapter

Text: ...དུ་བརྗོད་དོ། །གསང་བའི་སྙིང་པོ་དེ་ཁོ་ན་ཉིད་ངེས་པ་ལས་ཤིན་ཏུ་གསང་བ་མན་ངག་གི་སྙིང་པོའི་ལེའུ་སྟེ་བཅུ་གསུམ་པའོ།། །།
ལེའུ་བཅུ་བཞི་པ།
༈ ། །དོན་གསུམ་པ་འབྲས་བུ་སྐུ་དང་ཡེ་ཤེས་ཀྱི་རང་བཞིན་ལ་བསྟོད་པའི་ཚུལ་ལ་...

{"spans": [{"label": "CHAPTER", "head": "ལེའུ་བཅུ་བཞི་པ", "tail": ""}]}

14 characters, so it goes whole in head. The line above closes chapter thirteen and gets no
span. The line after the title is body text.

3 - a plain section title line, no opener, no number

Text: ...ཇོ་བོའི་ལུགས་ཀྱི་བསྔོ་བ་བྱའོ། །
སྨོན་པའི་ཆོ་ག་བསྡུས་པ།
སྨོན་པའི་ཆོ་ག་ལེན་ལུགས་རྒྱས་པ་ཐལ་ནས། ཤི་ཀ་མ་ལ་སོགས་པས་མི་ནུས་པས་མེ་ཏ...

{"spans": [{"label": "CHAPTER", "head": "སྨོན་པའི་ཆོ་ག་བསྡུས་པ", "tail": ""}]}

The line above ends a unit with །. The title stands alone on its line and the body starts on
the next one.

4 - front matter: a publisher's note heading

Text: ...མཛད་པ་པོ། དགའ་རབ་རྡོ་རྗེ། རྫ་དཔལ་སྤྲུལ་རིན་པོ་ཆེ། སངས་རྒྱས་མཉན་པ་རིན་པོ་ཆེ།
དཔེ་སྐྲུན་གསལ་བཤད།
ད་ལམ་འདིར་བན་ཆེན་གསུང་རབ་རྒྱུན་སྤེལ་ཁང་ནས་རིག་འཛིན་གྲུབ་པ་ཡོངས་ཀྱི་འ...

{"spans": [{"label": "CHAPTER", "head": "དཔེ་སྐྲུན་གསལ་བཤད", "tail": ""}]}

The author line above it is not part of the span.

5 - a short bare chapter-number line ending པའོ: it is a title, so it is marked

Text: ...འཁྲུལ་དུས་དེ་བཞིན་ཉིད་ལས་མ་གཡོས་པའི་ཚུལ། ཡེ་ནས་གྲོལ་བའི་ཆོས་སྟོན་ཚུལ་ལོ། །
ལེའུ་གཉིས་པའོ། །
གསུམ་པ་འཇིག་རྟེན་དུ་སྒྲོན་མ་བཀོད་པ་ལ་གསུམ་སྟེ། དོན་གྱི་འབྲེལ་འགོད། ཚིག་གི་དོན་བཤད། སྐབས་...

{"spans": [{"label": "CHAPTER", "head": "ལེའུ་གཉིས་པའོ", "tail": ""}]}

14 characters, so it goes whole in head. The line above is the last line of the previous
chapter's outline and gets no span. A long line that describes a chapter and then ends
...ལེའུ་སྟེ་བཅུ་གསུམ་པའོ།། །། is a closing colophon and is not marked.

6 - an outline heading and its body: a different layer, so nothing is marked

Text: ...བཞི་པ་༼དགོས་པ༽ནི།
བདག་ཉིད་ཆེན་པོ་རྣམས་ཀྱིས་ཆོས་ཀྱི་རྒྱལ་སྲིད་བསྒྲུབ་པ་ལ་རྒྱུ་ཡི་གཙོ་བོ་ཆོས་ཀྱི་རྒྱལ་པོ་ཉིད་ཡིན་པའི་ཕྱིར། འདི་ལ་བརྟེན་ནས་སྔགས་ཀྱི་ལམ་ཀུན་འབྱུང་ཞིང་ལམ་དང་འབྲས་བུ་ཐམས་ཅད་འདིའི་ངོ་བོ་ཉིད་དུ་གྱུར་པས་ན་ཤིན་ཏུ་གལ་ཆེ་བའོ། །མདོར་ན་དཀྱིལ་འཁོར་འདི་ནི། ལྟ་བས་ཤེས་པར་བྱ་བ་དང༌། བསྒོམ་པས་ཉམས་སུ་བླང་བར་བྱ་བའི་གཞིར་གྲུབ་ལ། སྤྱོད་པ...

{"spans": []}

བཞི་པ་༼དགོས་པ༽ནི། is an outline heading, which belongs to the sabche layer. The rest is
exposition. The right answer for a window like this is an empty list.

Output

Return the spans in the order they appear in the chunk. No offsets, no indices, no
translations, no explanations, no markdown fences.

Reply with JSON only: {"spans":[{"label":"CHAPTER","frame":"...","head":"...","tail":"..."}]}
frame is optional; head and tail are copied character for character. If the span is 40
characters or shorter, put it all in head and leave tail empty. If none: {"spans": []}

Text:

---
