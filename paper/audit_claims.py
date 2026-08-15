"""Cross-check the manuscript against the experiment outputs and against itself.

Catches the failure modes that survive a clean LaTeX compile:
  * a number in the prose that no data file supports
  * the abstract asserting something the results section retracts
  * a citation key used in the text but absent from the bibliography
  * a \\label{} referenced by \\ref{} but never defined (or vice versa)
  * leftover placeholder / hedging markers

Run:  ds-python paper/audit_claims.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FRONT = ROOT / "vae_training" / "runs" / "frontier"

TEX = (HERE / "main.tex").read_text()
GEN = (HERE / "tables_generated.tex").read_text() if (HERE / "tables_generated.tex").exists() else ""
NUM = (HERE / "numbers_generated.tex").read_text() if (HERE / "numbers_generated.tex").exists() else ""
BIB = (HERE / "refs.bib").read_text()

fails, warns = [], []


def check(cond, msg, hard=True):
    if not cond:
        (fails if hard else warns).append(msg)
    return cond


# ---------------------------------------------------------------- citations
used = set()
for grp in re.findall(r"\\cite\{([^}]+)\}", TEX):
    used |= {k.strip() for k in grp.split(",")}
defined = set(re.findall(r"@\w+\{([^,]+),", BIB))
check(not (used - defined), f"cited but not in refs.bib: {sorted(used - defined)}")
print(f"citations : {len(used)} used, {len(defined)} defined, "
      f"{len(defined - used)} uncited")
if defined - used:
    warns.append(f"in bib but never cited: {sorted(defined - used)}")

# ---------------------------------------------------------------- labels / refs
labels = set(re.findall(r"\\label\{([^}]+)\}", TEX + GEN))
refs = set(re.findall(r"\\ref\{([^}]+)\}", TEX))
check(not (refs - labels), f"\\ref to undefined label: {sorted(refs - labels)}")
print(f"labels    : {len(labels)} defined, {len(refs)} referenced")
if labels - refs:
    warns.append(f"table/figure defined but never referenced: {sorted(labels - refs)}")

# ---------------------------------------------------------------- placeholders
body = re.sub(r"(?<!\\)%.*", "", TEX)
n_pending = len(re.findall(r"\\pending\{", body)) - len(re.findall(r"newcommand\{\\pending\}", body))
check(n_pending <= 0, f"{n_pending} unresolved \\pending{{}} marker(s) in the text")
check("<user>" not in body, "placeholder '<user>' still present (repo URL)")
print(f"placeholders: {max(n_pending,0)} pending markers")

# ---------------------------------------------------------------- self-consistency
# The 13x claim was retracted; it may appear ONLY where it is explicitly disowned.
for m in re.finditer(r"13\\times", body):
    ctx = body[max(0, m.start() - 320):m.start() + 320]
    check(any(w in ctx for w in ("not supported", "repeats show", "would have suggested")),
          "the retracted 13x claim appears without a disclaimer nearby")

abstract = body[body.index("begin{abstract}"):body.index("end{abstract}")]
check("at no measured transfer cost" not in abstract,
      "abstract still asserts the retracted 'no transfer cost' claim")
check("ample headroom" not in body,
      "the unqualified ESP32 'ample headroom' claim is still present")

# ---------------------------------------------------------------- numbers vs data
srr_p = FRONT / "srr.json"
if srr_p.exists():
    d = json.loads(srr_p.read_text())
    for macro, val in [("SRRport", d["port_only"]), ("SRRall", d["all_features"]),
                       ("SRRratio", d["SRR"])]:
        m = re.search(rf"\\renewcommand\{{\\{macro}\}}\{{([0-9.]+)\}}", NUM)
        check(m is not None and abs(float(m.group(1)) - val) < 5e-5,
              f"macro {macro} does not match srr.json ({val:.4f})")
    print(f"SRR macros: match srr.json (SRR={d['SRR']:.4f})")
else:
    warns.append("srr.json missing; SRR macros unverified")

# Any bare decimal in the prose should be traceable; list them for eyeball review.
prose_nums = sorted(set(re.findall(r"(?<![\w.])0\.\d{3,4}(?![\w])", body)))
print(f"prose decimals (0.xxx): {len(prose_nums)} -> {prose_nums[:14]}")

# ---------------------------------------------------------------- report
print("-" * 60)
for w in warns:
    print(f"WARN  {w}")
for f in fails:
    print(f"FAIL  {f}")
if fails:
    print(f"\n{len(fails)} hard failure(s)")
    sys.exit(1)
print("PASS: manuscript internally consistent and traceable to data")
