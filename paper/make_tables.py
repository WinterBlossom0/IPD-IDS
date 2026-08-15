"""Generate paper/{tables,numbers}_generated.tex from the experiment CSVs.

main.tex \\input{}s both, so every number in the paper traces to a file produced by the
pipeline. Never hand-edit the generated files - rerun this instead.

Run:  ds-python paper/make_tables.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "vae_training" / "runs" / "frontier"
OUT = Path(__file__).resolve().parent / "tables_generated.tex"
NUMS = Path(__file__).resolve().parent / "numbers_generated.tex"

SIZE_ORDER = {"L": 0, "M": 1, "S": 2, "XS": 3}
SEED_BASE = 42


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def load():
    f = FRONT / "frontier.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    eff = FRONT / "efficiency.csv"
    if eff.exists():
        e = pd.read_csv(eff)[["tag", "size_name", "encoder_params", "int8_KB",
                              "cpu_latency_ms_1window"]]
        df = df.merge(e, on="tag", how="left")
    if "size_name" not in df.columns:
        return None
    df = df[df["size_name"].isin(SIZE_ORDER)].copy()
    df["_o"] = df["size_name"].map(SIZE_ORDER)
    if "seed" not in df.columns:
        df["seed"] = SEED_BASE
    df["seed"] = df["seed"].fillna(SEED_BASE)
    return df


def srr_table():
    p = FRONT / "srr.json"
    if not p.exists():
        return "% srr.json missing\n"
    d = json.loads(p.read_text())
    return f"""
\\begin{{table}}[t]
\\caption{{Shortcut Reliance Ratio on CSE-CIC-IDS2018. A depth-8 decision tree given only
the destination port recovers almost all of the score available from all 46 features.}}
\\label{{tab:srr}}
\\centering
\\begin{{tabular}}{{lc}}
\\toprule
Feature set & Binary $F_1$ \\\\
\\midrule
\\texttt{{Dst Port}} only & {d['port_only']:.4f} \\\\
All 46 features & {d['all_features']:.4f} \\\\
\\midrule
\\textbf{{SRR}} & \\textbf{{{d['SRR']:.4f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def grid_table(df):
    if df is None:
        return "% frontier.csv missing\n"
    g = df[df["seed"] == SEED_BASE].sort_values(["_o", "lam"])
    if g.empty:
        return ("\\begin{table}[t]\\caption{Grid (pending).}\\label{tab:grid}"
                "\\centering---\\end{table}\n")
    rows = []
    for _, r in g.iterrows():
        lam = "---" if r.get("drop_port") else f"{r['lam']:g}"
        note = "$^\\dagger$" if r.get("drop_port") else ""
        rows.append(
            (f"{esc(r['size_name'])}{note} & {lam} & "
             f"{int(r['encoder_params']):,} & "
             f"{r.get('mdl_linear_compression', np.nan):.2f} & "
             f"{r.get('mdl_mlp_compression', np.nan):.2f} & "
             f"{r.get('lgbm_test_purged_f1', np.nan):.3f} & "
             f"{r.get('mlp8_test_purged_f1', np.nan):.3f} \\\\").replace(",", "{,}"))
    return f"""
\\begin{{table}}[t]
\\caption{{Capacity $\\times$ erasure grid, seed 42. MDL compression is residual
destination-port information in the latent ($1.0 =$ none). $F_1$ is on the temporally
shifted, leakage-purged test split, under each of the two detection heads.
$^\\dagger$port column deleted instead of erased.}}
\\label{{tab:grid}}
\\centering
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{llrrrrr}}
\\toprule
 & & & \\multicolumn{{2}}{{c}}{{MDL compr.}} & \\multicolumn{{2}}{{c}}{{test $F_1$}} \\\\
\\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
Size & $\\lambda$ & Params & lin. & MLP & LGBM & MLP-8 \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def deploy_table(df):
    """Encoder + head + total. Quoting encoder size alone while scoring with LightGBM
    would understate the deployed system by three orders of magnitude."""
    if df is None:
        return ("\\begin{table}[t]\\caption{Deployed system cost (pending).}"
                "\\label{tab:deploy}\\centering---\\end{table}\n")
    g = df[(df["lam"] == 0) & (~df["drop_port"].astype(bool)) &
           (df["seed"] == SEED_BASE)].sort_values("_o")
    if "mlp8_head_KB" not in g.columns or g.empty or g["lgbm_head_KB"].isna().all():
        return ("\\begin{table}[t]\\caption{Deployed system cost (pending two-head "
                "evaluation).}\\label{tab:deploy}\\centering---\\end{table}\n")
    lk = float(g["lgbm_head_KB"].dropna().iloc[0])
    rows = []
    for _, r in g.iterrows():
        enc, mk = r["int8_KB"], r["mlp8_head_KB"]
        rows.append(
            (f"{esc(r['size_name'])} & {int(r['encoder_params']):,} & {enc:.2f} & "
             f"{enc + mk:.2f} & {r.get('mlp8_test_purged_f1', np.nan):.3f} & "
             f"{r.get('lgbm_test_purged_f1', np.nan):.3f} \\\\")
            .replace(",", "{,}"))
    return f"""
\\begin{{table}}[t]
\\caption{{Deployed system cost at $\\lambda=0$, int8. The encoder is small at every
capacity; the classifier is not. A LightGBM head adds a further ${lk:.0f}$~KB irrespective of
encoder size, so only the MLP-8 total describes a microcontroller-deployable system. The
decoder is training-only and never ships.}}
\\label{{tab:deploy}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lrrrcc}}
\\toprule
 & & \\multicolumn{{2}}{{c}}{{KB (enc.+MLP-8)}} & \\multicolumn{{2}}{{c}}{{test $F_1$}} \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}}
Size & Params & Enc. & Total & MLP-8 & LGBM \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def seed_table(df):
    """Error bars. With n=1 the between-size range sat inside the within-size spread."""
    if df is None:
        return "% frontier.csv missing\n"
    g = df[(df["lam"] == 0) & (~df["drop_port"].astype(bool))]
    if g.empty or g["seed"].nunique() < 2:
        return ("\\begin{table}[t]\\caption{Seed repeats (pending evaluation).}"
                "\\label{tab:seeds}\\centering---\\end{table}\n")
    rows = []
    for s in ["L", "M", "S", "XS"]:
        h = g[g["size_name"] == s]
        if h.empty:
            continue
        parts = [f"{esc(s)}", f"{int(h['encoder_params'].iloc[0]):,}", f"{len(h)}"]
        for head in ("lgbm", "mlp8"):
            c = f"{head}_test_purged_f1"
            parts.append(f"${h[c].mean():.3f} \\pm {h[c].std(ddof=1):.3f}$"
                         if c in h.columns and len(h) > 1 else "---")
        rows.append((" & ".join(parts) + " \\\\").replace(",", "{,}"))
    return f"""
\\begin{{table}}[t]
\\caption{{Capacity comparison with seed repeats at $\\lambda=0$: mean $\\pm$ sample standard
deviation of shifted-split $F_1$ over $n$ seeds. Overlapping intervals mean the capacity
ordering is not resolved by these runs.}}
\\label{{tab:seeds}}
\\centering
\\small
\\begin{{tabular}}{{lrrcc}}
\\toprule
Size & Params & $n$ & LGBM head & MLP-8 head \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def hybrid_table():
    """The two branches side by side. This is the system table: neither branch alone
    covers both known and novel attacks, and the numbers show why."""
    hp, op = FRONT / "hybrid.csv", FRONT / "ood.csv"
    if not (hp.exists() and op.exists()):
        return ("\\begin{table}[t]\\caption{Hybrid detector (pending).}"
                "\\label{tab:hybrid}\\centering---\\end{table}\n")
    h = pd.read_csv(hp); o = pd.read_csv(op)
    h = h[(h["lam"] == 0) & h["size_name"].isin(SIZE_ORDER)]
    o = o[(o["lam"] == 0) & o["size_name"].isin(SIZE_ORDER)]
    if h.empty or o.empty:
        return ("\\begin{table}[t]\\caption{Hybrid detector (pending).}"
                "\\label{tab:hybrid}\\centering---\\end{table}\n")
    hg = h.groupby("size_name").agg(n=("tag", "count"),
                                    params=("encoder_params", "first"),
                                    bin_f1=("lgbm_bin_f1", "mean"),
                                    mc=("lgbm_mc_attack_macro_f1", "mean"),
                                    rs=("lgbm_recall_seen", "mean"),
                                    ru=("lgbm_recall_unseen", "mean"))
    og = o.groupby("size_name").agg(auc=("auroc_recon_unseen", "mean"),
                                    aucsd=("auroc_recon_unseen", "std"),
                                    aseen=("auroc_recon_seen", "mean"))
    rows = []
    for k in ["L", "M", "S", "XS"]:
        if k not in hg.index or k not in og.index:
            continue
        a, b = hg.loc[k], og.loc[k]
        rows.append((f"{k} & {int(a['params']):,} & {a['bin_f1']:.3f} & {a['mc']:.3f} & "
                     f"{a['rs']:.3f} & {a['ru']:.3f} & "
                     f"${b['auc']:.3f} \\pm {b['aucsd']:.3f}$ & {b['aseen']:.3f} \\\\")
                    .replace(",", "{,}"))
    return f"""
\\begin{{table*}}[t]
\\caption{{The two branches of the detector, $\\lambda=0$, averaged over seeds. The
supervised head handles attack classes present in training and is near-blind to those that
are not; the reconstruction-error branch is the reverse. Attack macro-$F_1$ is over the
seen attack classes only. AUROC is threshold-free.}}
\\label{{tab:hybrid}}
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lrrrrrcc}}
\\toprule
 & & \\multicolumn{{4}}{{c}}{{Supervised branch (LightGBM head)}} & \\multicolumn{{2}}{{c}}{{Anomaly branch (recon.\\ error)}} \\\\
\\cmidrule(lr){{3-6}} \\cmidrule(lr){{7-8}}
Size & Params & bin.\\ $F_1$ & atk.\\ macro-$F_1$ & rec.\\ seen & rec.\\ unseen & AUROC unseen & AUROC seen \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def macros(df):
    out = []
    p = FRONT / "srr.json"
    if p.exists():
        d = json.loads(p.read_text())
        out += [f"\\renewcommand{{\\SRRport}}{{{d['port_only']:.4f}}}",
                f"\\renewcommand{{\\SRRall}}{{{d['all_features']:.4f}}}",
                f"\\renewcommand{{\\SRRratio}}{{{d['SRR']:.4f}}}"]
    if df is not None:
        base = df[(df["lam"] == 0) & (~df["drop_port"].astype(bool)) &
                  (df["seed"] == SEED_BASE)].drop_duplicates("size_name").set_index("size_name")
        for k, m in [("L", "paramsL"), ("M", "paramsM"), ("S", "paramsS"), ("XS", "paramsXS")]:
            if k in base.index:
                out.append((f"\\renewcommand{{\\{m}}}{{{int(base.loc[k, 'encoder_params']):,}}}")
                           .replace(",", "{,}"))
        for k, m in [("S", "totalS"), ("XS", "totalXS")]:
            if k in base.index and "mlp8_head_KB" in base.columns:
                out.append(f"\\renewcommand{{\\{m}}}"
                           f"{{{base.loc[k, 'int8_KB'] + base.loc[k, 'mlp8_head_KB']:.2f}}}")
        if "lgbm_head_KB" in base.columns and len(base):
            out.append(f"\\renewcommand{{\\lgbmKB}}{{{base['lgbm_head_KB'].iloc[0]:.0f}}}")
    return "\n".join(out) + "\n"


def main():
    df = load()
    NUMS.write_text("% AUTO-GENERATED by paper/make_tables.py - do not edit by hand.\n"
                    + macros(df))
    OUT.write_text("\n".join([
        "% AUTO-GENERATED by paper/make_tables.py - do not edit by hand.\n",
        srr_table(), hybrid_table(), grid_table(df), seed_table(df),
        deploy_table(df)]))
    print(f"wrote {NUMS}\nwrote {OUT}")
    if df is not None:
        print(f"  rows: {len(df)}  seeds: {sorted(df['seed'].unique().tolist())}")
        print(f"  heads: {[c for c in df.columns if c.endswith('_head_KB')]}")


if __name__ == "__main__":
    main()
