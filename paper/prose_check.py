"""Flag AI-writing tells in main.tex, following Wikipedia:Signs of AI writing.

The point is not that these constructions are wrong. It is that they cluster in
LLM-generated prose, and a reviewer who reads a lot of submissions notices the cluster.
Everything here is a prompt to re-read a sentence, not an automatic edit.

Run:  python3 paper/prose_check.py
"""
import re
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent / "main.tex"

# Strip comments, math, macros and the bibliography so we lint prose only.
raw = TEX.read_text()
body = re.sub(r"(?<!\\)%.*", "", raw)
body = re.sub(r"\$[^$]*\$", " NUM ", body)
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", body)
body = re.sub(r"\s+", " ", body)

CHECKS = [
    ("em dash",            r"---",                                              8),
    ("AI vocabulary",      r"\b(delve|underscore[sd]?|pivotal|crucial|intricac\w+|"
                           r"testament|tapestry|vibrant|foster\w*|bolster\w*|"
                           r"seamless\w*|realm|landscape of|robustly)\b",        0),
    ("negative parallel",  r"\bnot (just|only|merely)\b[^.]{0,60}\bbut\b",       0),
    ("copula avoidance",   r"\b(serves as|functions as|stands as|acts as)\b",    0),
    ("marketing verb",     r"\b(boasts|showcases|leverages|harnesses)\b",        0),
    ("significance puff",  r"\b(plays? a (vital|key|critical) role|"
                           r"marking a \w+ moment|highlights the importance|"
                           r"paving the way|sets? the stage)\b",                 0),
    ("trailing -ing gloss", r", (highlighting|demonstrating|showcasing|"
                            r"underscoring|reflecting|emphasi[sz]ing)\b",        2),
    ("vague attribution",  r"\b(studies show|experts (say|argue)|it is widely|"
                           r"observers have noted|industry reports)\b",          0),
    ("summary filler",     r"\b(In summary|Overall,|In conclusion|It is worth noting"
                           r"|It should be noted)\b",                            0),
    ("hedge stack",        r"\b(may potentially|could possibly|might perhaps)\b", 0),
]

print(f"prose length: {len(body):,} chars\n")
total_over = 0
for name, pat, budget in CHECKS:
    hits = re.findall(pat, body, flags=re.I)
    n = len(hits)
    flag = "OK " if n <= budget else "OVER"
    if n > budget:
        total_over += 1
    extra = ""
    if n and name != "em dash":
        uniq = sorted({(h if isinstance(h, str) else h[0]).lower() for h in hits})[:6]
        extra = "  " + ", ".join(uniq)
    print(f"  {flag} {name:<20} {n:>3}  (budget {budget}){extra}")

# Sentence length spread: uniform length is itself a tell.
sents = [s for s in re.split(r"(?<=[.!?]) ", body) if len(s) > 25]
if sents:
    lens = [len(s.split()) for s in sents]
    mean = sum(lens) / len(lens)
    var = (sum((l - mean) ** 2 for l in lens) / len(lens)) ** 0.5
    print(f"\n  sentences: {len(sents)}  mean {mean:.1f} words  sd {var:.1f}"
          f"   ({'varied' if var > 7 else 'UNIFORM - reads mechanical'})")

print("\n" + ("-" * 52))
print("clean" if total_over == 0 else f"{total_over} category(ies) over budget")
sys.exit(0)
