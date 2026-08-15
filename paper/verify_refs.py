"""Rebuild refs.bib from AUTHORITATIVE records only. No search summaries, no guesses.

Sources, in order of preference:
  1. DBLP  - canonical for CS venues. We take DBLP's own BibTeX export, so authors,
             venue, pages and DOI are exactly as indexed rather than retyped.
  2. arXiv API - for preprints with no venue record; the Atom feed IS the submission record.

Every candidate must clear TWO hard gates before it is written:
  * title similarity >= 0.85 against the expected title, and
  * publication year within +/-1 of the expected year.

The year gate is not paranoia. Crossref and OpenAlex both return a 2025 "posted-content"
record for Attention Is All You Need (a 2017 NeurIPS paper); an unvalidated pipeline would
have silently written that into the bibliography. Anything failing either gate is DROPPED
and named in the report, so an omission is always a visible decision.

DBLP rate-limits aggressively: requests are paced and backed off, which makes a full run
slow on purpose.

Run:  ds-python paper/verify_refs.py
"""
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "refs.bib"
CACHE = HERE / ".refcache.json"
UA = {"User-Agent": "refs-verifier/2.0 (academic bibliography verification)"}

TITLE_MIN = 0.85
YEAR_TOL = 1
DBLP_PACE = 4.0

# (citekey, expected title, expected year, arxiv_id or None)
WANTED = [
    ("liu2022error", "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018", 2022, None),
    ("engelen2021troubleshooting", "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study", 2021, None),
    ("sharafaldin2018toward", "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 2018, None),
    ("goldschmidt2025datasets", "Network Intrusion Datasets: A Survey, Limitations, and Recommendations", 2025, "2502.06688"),
    ("cantone2024crossdataset", "On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection", 2024, "2402.10974"),
    ("arp2022dos", "Dos and Don'ts of Machine Learning in Computer Security", 2022, None),
    ("pendlebury2019tesseract", "TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time", 2019, None),
    ("geirhos2020shortcut", "Shortcut learning in deep neural networks", 2020, None),
    ("kumar2022probing", "Probing Classifiers are Unreliable for Concept Removal and Detection", 2022, None),
    ("belrose2023leace", "LEACE: Perfect linear concept erasure in closed form", 2023, None),
    ("voita2020mdl", "Information-Theoretic Probing with Minimum Description Length", 2020, None),
    ("ganin2016dann", "Domain-Adversarial Training of Neural Networks", 2016, None),
    ("layeghy2022dinids", "DI-NIDS: Domain Invariant Network Intrusion Detection System", 2022, "2210.08252"),
    ("roy2023domaininvariant", "Improving Intrusion Detection with Domain-Invariant Representation Learning in Latent Space", 2023, "2312.17300"),
    ("kingma2014vae", "Auto-Encoding Variational Bayes", 2014, "1312.6114"),
    ("higgins2017betavae", "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", 2017, None),
    ("kingma2016iaf", "Improving Variational Inference with Inverse Autoregressive Flow", 2016, "1606.04934"),
    ("saha2025ardvae", "ARD-VAE: A Statistical Formulation to Find the Relevant Latent Dimensions of Variational Autoencoders", 2025, "2501.10901"),
    ("vaswani2017attention", "Attention Is All You Need", 2017, "1706.03762"),
    ("lo2022egraphsage", "E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT", 2022, None),
    ("ravfogel2020inlp", "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection", 2020, None),
    ("arjovsky2019irm", "Invariant Risk Minimization", 2019, "1907.02893"),
    ("moustafa2015unsw", "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)", 2015, None),
    ("sarhan2021netflow", "NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems", 2021, None),
    ("zavrak2020vae", "Anomaly-Based Intrusion Detection From Network Flow Features Using Variational Autoencoder", 2020, None),
    ("deepseekv2", "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", 2024, "2405.04434"),
    ("beltiukov2025demystifying", "Demystifying Network Foundation Models", 2025, "2509.23089"),
    ("elmahdaouy2026contextualized", "Deep Learning for Contextualized NetFlow-Based Network Intrusion Detection: Methods, Data, Evaluation and Deployment", 2026, "2602.05594"),
    ("wang2026bias", "Bias in the Shadows: Explore Shortcuts in Encrypted Network Traffic Classification", 2026, "2601.10180"),
    ("hakim2026crossdomain", "Cross-Domain Generalization Failure in Lightweight Intrusion Detection Models for IIoT Networks", 2026, "2607.00553"),
    ("abushahla2025quantization", "Neural Network Quantization for Microcontrollers: A Comprehensive Survey of Methods, Platforms, and Applications", 2025, "2508.15008"),
    ("guthula2024netfound", "netFound: Foundation Model for Network Security", 2023, "2310.17025"),
]


def get(url, tries=5, base=3.0):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            code = getattr(e, "code", None)
            wait = base * (2 ** i) if code == 429 or code is None else base
            if i < tries - 1:
                time.sleep(min(wait, 45))
    return None


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def sim(a, b):
    return difflib.SequenceMatcher(None, " ".join(norm(a).split()),
                                   " ".join(norm(b).split())).ratio()


def dblp_lookup(title, year):
    raw = get(f"https://dblp.org/search/publ/api?q={urllib.parse.quote(title)}&format=json&h=8")
    if not raw:
        return None, "dblp unreachable"
    try:
        hits = json.loads(raw)["result"]["hits"].get("hit", [])
    except Exception:
        return None, "dblp parse error"
    if isinstance(hits, dict):
        hits = [hits]
    best, best_r, why = None, 0.0, "no hit above title threshold"
    for h in hits:
        info = h.get("info", {})
        r = sim(title, info.get("title", ""))
        try:
            y = int(info.get("year", 0))
        except ValueError:
            y = 0
        if r < TITLE_MIN:
            continue
        if abs(y - year) > YEAR_TOL:
            why = f"year mismatch (got {y}, expected {year})"
            continue
        if info.get("venue") == "CoRR":
            r -= 0.05          # prefer the venue record over the arXiv mirror
        if r > best_r:
            best, best_r = info, r
    return (best, best_r) if best else (None, why)


def dblp_bibtex(key):
    return get(f"https://dblp.org/rec/{key}.bib?param=1")


def arxiv_lookup(aid, title, year):
    raw = get(f"http://export.arxiv.org/api/query?id_list={aid}")
    if not raw:
        return None, "arxiv unreachable"
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        e = ET.fromstring(raw).find("a:entry", ns)
        if e is None:
            return None, "arxiv no entry"
        got = " ".join(e.find("a:title", ns).text.split())
        authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
        y = int(e.find("a:published", ns).text[:4])
    except Exception:
        return None, "arxiv parse error"
    if sim(title, got) < TITLE_MIN:
        return None, f"arxiv title mismatch ({got[:50]!r})"
    if abs(y - year) > YEAR_TOL:
        return None, f"arxiv year mismatch (got {y}, expected {year})"
    return {"title": got, "authors": authors, "year": y}, "ok"


def bib_from_arxiv(key, aid, rec):
    return (f"@article{{{key},\n"
            f"  author        = {{{' and '.join(rec['authors'])}}},\n"
            f"  title         = {{{rec['title']}}},\n"
            f"  journal       = {{arXiv preprint arXiv:{aid}}},\n"
            f"  eprint        = {{{aid}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  year          = {{{rec['year']}}}\n"
            f"}}\n")


def main():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    report = []

    for key, title, year, aid in WANTED:
        if key in cache:
            report.append((key, cache[key]["src"], cache[key]["rec"]))
            continue

        entry = src = rec = None
        info, why = dblp_lookup(title, year)
        time.sleep(DBLP_PACE)
        if info:
            bib = dblp_bibtex(info["key"])
            time.sleep(DBLP_PACE)
            if bib and bib.strip().startswith("@"):
                bib = re.sub(r"@(\w+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{key},", bib, count=1)
                bib = re.sub(r"^\s*(biburl|bibsource|timestamp)\s*=.*$", "", bib, flags=re.M)
                bib = re.sub(r",\s*\n\s*\n*\}", "\n}", bib)
                bib = re.sub(r"\n{2,}", "\n", bib).strip()
                entry = (f"% [DBLP {info.get('venue','?')} {info.get('year','?')}] "
                         f"title match {sim(title, info.get('title','')):.2f} | {info.get('url','')}\n"
                         + bib + "\n")
                src, rec = "DBLP", f"{info.get('venue','?')} {info.get('year','?')}"

        if entry is None and aid:
            arec, awhy = arxiv_lookup(aid, title, year)
            time.sleep(1.0)
            if arec:
                entry = (f"% [arXiv API] submission record arXiv:{aid}, "
                         f"title match {sim(title, arec['title']):.2f}\n"
                         + bib_from_arxiv(key, aid, arec))
                src, rec = "arXiv", f"arXiv:{aid} ({arec['year']})"
            else:
                why = f"{why}; {awhy}"

        if entry:
            cache[key] = {"bib": entry, "src": src, "rec": rec}
            report.append((key, src, rec))
        else:
            report.append((key, "DROPPED", why))
        CACHE.write_text(json.dumps(cache, indent=1))
        print(f"  {key:<32}{report[-1][1]:<9}{report[-1][2]}", flush=True)

    header = ("% ============================================================================\n"
              "% Bibliography. EVERY entry was resolved against an authoritative record and\n"
              "% passed two hard gates: title similarity >= 0.85 and year within +/-1.\n"
              "%   [DBLP ...]  - DBLP's own BibTeX export (authors/venue/pages/DOI as indexed)\n"
              "%   [arXiv API] - the arXiv submission record itself\n"
              "% No entry here derives from a search-result summary. Unresolvable references\n"
              "% were dropped rather than guessed. Regenerate: ds-python paper/verify_refs.py\n"
              "% ============================================================================\n")
    body = "\n".join(cache[k]["bib"] for k, _, _, _ in WANTED if k in cache)
    OUT.write_text(header + "\n" + body)

    ok = sum(1 for _, s, _ in report if s != "DROPPED")
    print("-" * 78)
    print(f"resolved {ok}/{len(report)} -> {OUT}")
    dropped = [(k, d) for k, s, d in report if s == "DROPPED"]
    for k, d in dropped:
        print(f"  DROPPED {k}: {d}")
    return 0 if ok >= 20 else 1


if __name__ == "__main__":
    sys.exit(main())
