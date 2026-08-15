"""Prove the template change did not alter content.

Compares the body prose of two rendered PDFs sentence by sentence, ignoring the parts a
template legitimately changes: front-matter layout, section numbering style, float
labelling, hyphenation, and bibliography formatting.

Usage:  /tmp/pdfenv/bin/python paper/diff_versions.py OLD.pdf NEW.pdf
"""
import difflib
import re
import sys

import pypdf


def body_sentences(path):
    t = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
    # cut the bibliography: formatting differs by design between IEEEtran and elsarticle
    for marker in ("References", "REFERENCES"):
        i = t.rfind(marker)
        if i > len(t) * 0.5:
            t = t[:i]
            break
    t = re.sub(r"-\n", "", t)            # undo hyphenation at line breaks
    t = re.sub(r"\s+", " ", t)
    # normalise template-specific labelling
    t = re.sub(r"\b(?:I{1,3}|IV|V|VI{1,3}|IX|X)\.\s+(?=[A-Z])", "", t)   # roman headings
    t = re.sub(r"\b\d{1,2}\.\s+(?=[A-Z][a-z])", "", t)                    # arabic headings
    t = re.sub(r"\bTABLE\s+[IVX]+\b", "TABLE", t)
    t = re.sub(r"\bTable\s+[IVX\d]+\b", "Table", t)
    t = re.sub(r"\bFig\.\s*\d+\.?", "Fig.", t)
    t = re.sub(r"\[\d+(?:,\s*\d+)*\]", "[CITE]", t)                       # citation numbers
    t = t.replace("ﬀ", "ff").replace("ﬁ", "fi").replace("ﬃ", "ffi")
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = t.replace("—", "-").replace("–", "-").replace("−", "-")
    # Column width differs between templates, so PDF extraction breaks words and numbers
    # in different places ("0 .9257" vs "0.9257", "withindataset" vs "within-dataset").
    # Compare a normalised word stream; that is the content, independent of line breaks.
    t = re.sub(r"(?<=\d)\s+(?=[.,]\d)", "", t)      # rejoin split decimals
    t = re.sub(r"(?<=[.,])\s+(?=\d)", "", t)
    t = re.sub(r"[^A-Za-z0-9.,%()/\[\]<>=+±→×∈∼-]+", " ", t)
    words = [w for w in t.split() if w not in ("-", ".", ",")]
    return words


old_p, new_p = sys.argv[1], sys.argv[2]
old, new = body_sentences(old_p), body_sentences(new_p)
print(f"{old_p}: {len(old)} body words")
print(f"{new_p}: {len(new)} body words\n")

sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
removed, added = [], []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag in ("delete", "replace"):
        removed += old[i1:i2]
    if tag in ("insert", "replace"):
        added += new[j1:j2]

print(f"similarity: {sm.ratio():.4f}")
print(f"words only in OLD: {len(removed)}")
print(f"words only in NEW: {len(added)}\n")

for lbl, xs in (("ONLY IN OLD (potentially lost)", removed), ("ONLY IN NEW (added)", added)):
    if xs:
        print(f"--- {lbl} ---")
        print("  " + " | ".join(xs[:30]))
        if len(xs) > 30:
            print(f"  ... and {len(xs)-30} more")
        print()

# numeric content must be identical regardless of template
def nums(ss):
    return sorted(re.findall(r"\d+\.\d+|\d{1,3}(?:,\d{3})+", " ".join(ss)))
no, nn = nums(old), nums(new)
missing = [n for n in set(no) if no.count(n) > nn.count(n)]
print("numeric content identical:" , "YES" if not missing else f"NO - missing {sorted(missing)}")
sys.exit(0 if not removed and not missing else 1)
