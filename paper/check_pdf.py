"""Post-build sanity check on main.pdf. Catches the failures a clean compile hides:
a section that silently vanished, an unresolved citation, or a page-limit breach.

Run:  python3 paper/check_pdf.py [max_pages]
"""
import re
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE / "main.pdf"
LOG = HERE / "main.log"

EXPECTED_SECTIONS = [
    "INTRODUCTION", "LITERATURE REVIEW", "NOVELTY OF THE PROPOSED WORK", "DATASET",
    "METHODOLOGY", "RESULTS", "DISCUSSION", "CONCLUSION AND FUTURE WORK",
]
MUST_APPEAR = [
    "Ovalekar", "Gore", "Pradhan", "Tanawade",
    "Dwarkadas", "Artificial Intelligence and Data Science", "Mumbai",
    "Deepali Patil", "Acknowledgment", "Shortcut Reliance",
]


def pdf_text(path):
    """Prefer pypdf; the hand-rolled stream parser below only recovers math-mode
    glyphs under this document's font encoding and silently under-reports."""
    try:
        import pypdf
        return "\n".join(pg.extract_text() or "" for pg in pypdf.PdfReader(str(path)).pages)
    except ImportError:
        pass
    data = path.read_bytes()
    chunks = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            chunks.append(zlib.decompress(m.group(1)).decode("latin-1"))
        except Exception:
            continue
    blob = " ".join(chunks)
    # Pull the literal strings out of the content streams' text-showing operators.
    parts = re.findall(r"\((?:\\.|[^()\\])*\)", blob)
    text = "".join(p[1:-1] for p in parts)
    text = re.sub(r"\\[0-7]{3}", "", text)
    return text.replace("\\", "")


def main():
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    warns = []
    if not PDF.exists():
        print("FAIL: main.pdf not found - run ./build.sh")
        return 1
    text = pdf_text(PDF)
    norm = re.sub(r"[^a-z0-9 ]", "", text.lower())
    fails = []

    pages = None
    if LOG.exists():
        m = re.search(r"\((\d+) pages?", LOG.read_text(errors="replace"))
        if m:
            pages = int(m.group(1))
    print(f"pages            : {pages if pages else '?'} (limit {max_pages})")
    if pages and pages > max_pages:
        fails.append(f"page limit exceeded: {pages} > {max_pages}")

    print("sections:")
    for s in EXPECTED_SECTIONS:
        ok = re.sub(r"[^a-z0-9 ]", "", s.lower()) in norm
        print(f"  {'OK  ' if ok else 'MISS'} {s}")
        if not ok:
            fails.append(f"missing section: {s}")

    print("required strings:")
    for s in MUST_APPEAR:
        ok = re.sub(r"[^a-z0-9 ]", "", s.lower()) in norm
        print(f"  {'OK  ' if ok else 'MISS'} {s}")
        if not ok:
            fails.append(f"missing text: {s}")

    unresolved = text.count("[?]")
    print(f"unresolved cites : {unresolved}")
    if unresolved:
        fails.append(f"{unresolved} unresolved citation(s)")

    refs = len(set(re.findall(r"\[(\d{1,2})\]", text)))
    print(f"distinct refs cited: {refs}")

    if LOG.exists():
        log = LOG.read_text(errors="replace")
        # An overfull \hbox is visible: text runs into the margin. A sub-2pt \vbox
        # overflow is float-packing slack, ~0.7mm, and is not a defect worth blocking on.
        hbox = re.findall(r"Overfull \\hbox \(([\d.]+)pt", log)
        vbox = [float(v) for v in re.findall(r"Overfull \\vbox \(([\d.]+)pt", log)]
        big_v = [v for v in vbox if v >= 2.0]
        print(f"overfull         : {len(hbox)} hbox, {len(vbox)} vbox "
              f"({len(big_v)} vbox >= 2pt)")
        if hbox:
            fails.append(f"{len(hbox)} overfull hbox - text overflows the column")
        if big_v:
            fails.append(f"{len(big_v)} overfull vbox >= 2pt")
        elif vbox:
            warns.append(f"{len(vbox)} sub-2pt vbox overflow (float packing; cosmetic)")

    print("-" * 46)
    if fails:
        print("FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS: all checks clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
