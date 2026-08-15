"""Whole-manuscript consistency sweep, run against the rendered PDF.

Checks things the other two gates do not: float citation order, duplicated sentences left
behind by editing, numbers that appear in the prose but in no data file, section ordering,
and abstract structure.

Run:  /tmp/pdfenv/bin/python paper/final_review.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pypdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FRONT = ROOT / "vae_training" / "runs" / "frontier"

pdf = pypdf.PdfReader(str(HERE / "main.pdf"))
T = "\n".join(p.extract_text() or "" for p in pdf.pages)
flat = re.sub(r"\s+", " ", T)
fails, warns = [], []

print(f"pages: {len(pdf.pages)}\n")

# ---- 1. float citation order
caps = [(m.start(), "Fig", m.group(1)) for m in re.finditer(r"Fig\.\s*(\d+)\.\s+[A-Z]", T)]
tcaps = [(m.start(), "Table", m.group(1)) for m in re.finditer(r"Table\s+(\d+)\s*[:.]?\s*[A-Z]", T)]
refs = [(m.start(), "Fig", m.group(1)) for m in re.finditer(r"Fig\.\s*(\d+)(?![\.\d])", T)]
trefs = [(m.start(), "Table", m.group(1)) for m in re.finditer(r"Table\s+(\d+)(?!\w)", T)]
seen, order = set(), []
for pos, typ, num in sorted(caps + tcaps + refs + trefs):
    if (typ, num) not in seen and (pos, typ, num) in refs + trefs:
        seen.add((typ, num))
        order.append(f"{typ} {num}")
figs = [o for o in order if o.startswith("Fig")]
tabs = [o for o in order if o.startswith("Table")]
print("float first-citation order:", " -> ".join(order))
if figs != sorted(figs, key=lambda x: int(x.split()[1])):
    fails.append(f"figures cited out of order: {figs}")
roman = {str(i): i for i in range(1, 9)}
if tabs != sorted(tabs, key=lambda x: roman.get(x.split()[1], 99)):
    fails.append(f"tables cited out of order: {tabs}")

# ---- 2. section order
# Match the NUMBERED heading, not the bare word: "Results" also appears as an abstract
# structure label, and matching that gives a false ordering failure.
want = ["INTRODUCTION", "LITERATURE REVIEW", "NOVELTY", "DATASET", "METHODOLOGY",
        "RESULTS", "DISCUSSION", "CONCLUSION"]
roman_pre = r"(?:\d{1,2}|I{1,3}|IV|V|VI{1,3}|IX|X)\."
pos = []
for w in want:
    m = re.search(roman_pre + r"\s*" + w.title().replace(" ", r"\s+"), T, re.I)
    pos.append((m.start() if m else -1, w))
if any(p < 0 for p, _ in pos):
    fails.append(f"missing section: {[w for p, w in pos if p < 0]}")
elif [w for _, w in sorted(pos)] != want:
    fails.append(f"sections out of order: {[w for _, w in sorted(pos)]}")
else:
    print("sections: all 8 present and in order")

# ---- 3. abstract structure
abs_i = flat.find("Abstract")
abs_txt = flat[abs_i:abs_i + 2200]
# The opening paragraph carries the problem statement unlabelled; the remaining
# three sections are labelled.
want_lbl = ("Method", "Results", "Conclusion")
labels = [l for l in want_lbl if l in abs_txt]
print(f"abstract structure labels: {labels}")
if len(labels) < len(want_lbl):
    warns.append(f"abstract missing structure labels: {set(want_lbl) - set(labels)}")

# ---- 4. duplicated sentences (editing leftovers)
sents = [s.strip() for s in re.split(r"(?<=[.!?]) ", flat) if len(s.strip()) > 70]
dupes = [s for s, c in Counter(sents).items() if c > 1]
print(f"sentences >70 chars: {len(sents)}, duplicated: {len(dupes)}")
for d in dupes[:3]:
    fails.append(f"duplicated sentence: {d[:70]}...")

# ---- 5. every prose decimal traceable to a data file
data_blob = ""
for f in ("srr.json",):
    if (FRONT / f).exists():
        data_blob += json.dumps(json.loads((FRONT / f).read_text()))
for f in ("frontier.csv", "efficiency.csv", "hybrid.csv", "ood.csv"):
    if (FRONT / f).exists():
        data_blob += (FRONT / f).read_text()
prose_nums = set(re.findall(r"(?<![\w.])0\.\d{3}(?![\d])", flat))
untraceable = []
data_vals = [float(x) for x in re.findall(r"\d+\.\d+", data_blob)]
for n in sorted(prose_nums):
    v = float(n)
    # prose rounds to 3dp; a CSV value that rounds to the same 3dp is the same number
    if not any(abs(round(d, 3) - v) < 1e-9 for d in data_vals):
        untraceable.append(n)
print(f"prose decimals: {len(prose_nums)}, not found verbatim in data files: "
      f"{len(untraceable)}")
if untraceable:
    warns.append(f"decimals not verbatim in CSVs (may be rounded/derived): {untraceable}")

# ---- 6. front matter
for probe in ("Ovalekar", "Gore", "Pradhan", "Tanawade", "djsce.edu.in",
              "Deepali Patil", "Artificial Intelligence and Data Science"):
    if probe.lower() not in flat.lower():
        fails.append(f"front matter missing: {probe}")
print("front matter: authors, emails, dept, mentor all present"
      if not any("front matter" in f for f in fails) else "front matter INCOMPLETE")

print("\n" + "-" * 58)
for w in warns:
    print(f"WARN  {w}")
for f in fails:
    print(f"FAIL  {f}")
print("PASS: no inconsistencies found" if not fails else f"{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
