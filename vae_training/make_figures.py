"""Publication figures. IEEE two-column: 3.4in single column, 7.0in full width.

Four figures, each carrying one claim the paper makes:

  fig1  srr           the hook - one field recovers almost the whole benchmark score
  fig2  complementary the system argument - each branch covers what the other misses
  fig3  erasure       erasure is verified, and residual port info falls with lambda
  fig4  deployment    where the bytes actually go (the head, not the encoder)

An earlier version drew one polyline through points from different model capacities,
producing meaningless zig-zag. Capacity is an entity, not a position on a curve: it gets a
colour, and lambda moves along its line.

Run:  ds-python make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent / "runs"
FRONT = RUNS / "frontier"
FIGS = RUNS / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.edgecolor": "#c3ccd3", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#d9dfe4", "grid.alpha": 0.7,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

ORDER = ["L", "M", "S", "XS"]
COLOR = {"L": "#2a78d6", "M": "#eb6834", "S": "#1baf7a", "XS": "#eda100"}
MARK = {"L": "o", "M": "s", "S": "^", "XS": "D"}
INK, MUTED, RULE = "#0d1117", "#79838b", "#c3ccd3"
SUP, ANO = "#b8262b", "#0c6553"


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


# ------------------------------------------------------------------ fig 1
def fig_srr():
    d = json.loads((FRONT / "srr.json").read_text())
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    names = ["all 46 features", "Dst Port alone"]
    vals = [d["all_features"], d["port_only"]]
    cols = ["#235b7d", SUP]
    bars = ax.barh(names, vals, height=0.5, color=cols, zorder=3)
    for r, v in zip(bars, vals):
        ax.text(v - 0.004, r.get_y() + r.get_height() / 2, f"{v:.3f}", va="center",
                ha="right", fontsize=7.5, color="white", fontweight="bold")
    ax.set_xlim(0.80, 0.98)
    ax.set_xlabel("binary $F_1$ (depth-8 decision tree)")
    ax.set_title(f"One field recovers {d['SRR']:.1%} of the score", pad=6)
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_srr")


# ------------------------------------------------------------------ fig 2
def fig_complementary():
    """The system argument, and the paper's most important figure."""
    h = pd.read_csv(FRONT / "hybrid.csv")
    o = pd.read_csv(FRONT / "ood.csv")
    h = h[(h["lam"] == 0) & h["size_name"].isin(ORDER)]
    o = o[(o["lam"] == 0) & o["size_name"].isin(ORDER)]
    hg = h.groupby("size_name").agg(rs=("lgbm_recall_seen", "mean"),
                                    ru=("lgbm_recall_unseen", "mean"))
    og = o.groupby("size_name").agg(au=("auroc_recon_unseen", "mean"),
                                    ausd=("auroc_recon_unseen", "std"),
                                    as_=("auroc_recon_seen", "mean"))
    sizes = [s for s in ORDER if s in hg.index and s in og.index]
    x = np.arange(len(sizes)); w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.8))

    ax = axes[0]
    ax.bar(x - w / 2, [hg.loc[s, "rs"] for s in sizes], w, color=SUP, zorder=3,
           label="attack classes seen in training")
    ax.bar(x + w / 2, [hg.loc[s, "ru"] for s in sizes], w, color=SUP, alpha=0.35,
           zorder=3, label="attack classes never seen")
    for i, s in enumerate(sizes):
        ax.text(i + w / 2, hg.loc[s, "ru"] + 0.02, f"{hg.loc[s,'ru']:.3f}",
                ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("recall")
    ax.set_xlabel("encoder capacity")
    ax.set_title("Supervised branch", color=SUP, pad=6)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.errorbar(x, [og.loc[s, "au"] for s in sizes],
                yerr=[og.loc[s, "ausd"] for s in sizes], fmt="o-", color=ANO,
                capsize=2.5, zorder=3, label="attack classes never seen")
    ax.plot(x, [og.loc[s, "as_"] for s in sizes], "s--", color=ANO, alpha=0.45,
            zorder=3, label="attack classes seen in training")
    ax.axhline(0.5, ls=":", lw=1.0, color=MUTED)
    ax.text(len(sizes) - 0.5, 0.505, "chance", ha="right", va="bottom",
            fontsize=6.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylim(0.25, 1.03)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("encoder capacity")
    ax.set_title("Anomaly branch (reconstruction error)", color=ANO, pad=6)
    ax.legend(frameon=False, loc="center right")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Each branch covers what the other misses", fontsize=9, y=1.04)
    save(fig, "fig2_complementary")


# ------------------------------------------------------------------ fig 3
def fig_erasure():
    df = pd.read_csv(FRONT / "frontier.csv")
    eff = pd.read_csv(FRONT / "efficiency.csv")[["tag", "size_name", "encoder_params"]]
    df = df.merge(eff, on="tag", how="left")
    df = df[df["size_name"].isin(ORDER) & (~df["drop_port"].astype(bool))]
    if "seed" in df.columns:
        df = df[df["seed"].fillna(42) == 42]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for s in ORDER:
        g = df[df["size_name"] == s].sort_values("lam")
        if g.empty:
            continue
        ax.plot(g["lam"], g["mdl_mlp_compression"], MARK[s] + "-", color=COLOR[s],
                label=f"{s} ({int(g['encoder_params'].iloc[0]):,})", zorder=3)
    ax.axhline(1.0, ls=":", lw=1.0, color=MUTED)
    ax.text(1.0, 1.02, "no port information", ha="right", va="bottom",
            fontsize=6.5, color=MUTED)
    ax.set_xlabel(r"erasure strength $\lambda$")
    ax.set_ylabel("residual port information\n(MDL compression)")
    ax.set_title("Erasure verified by a held-out probe", pad=6)
    ax.legend(frameon=False, title="capacity (params)", title_fontsize=6.5)
    save(fig, "fig3_erasure")


# ------------------------------------------------------------------ fig 4
def fig_deployment():
    """Grouped, not stacked. Stacking is additive and a log axis is multiplicative, so a
    stacked bar on a log scale is not readable as a sum - an earlier version of this figure
    made that mistake and lost the XS encoder bar entirely."""
    df = pd.read_csv(FRONT / "frontier.csv")
    eff = pd.read_csv(FRONT / "efficiency.csv")[["tag", "size_name", "encoder_params", "int8_KB"]]
    df = df.merge(eff, on="tag", how="left")
    df = df[(df["lam"] == 0) & df["size_name"].isin(ORDER) & (~df["drop_port"].astype(bool))]
    if "seed" in df.columns:
        df = df[df["seed"].fillna(42) == 42]
    if "mlp8_head_KB" not in df.columns:
        print("  (fig4 skipped: two-head data absent)")
        return
    df = df.drop_duplicates("size_name").set_index("size_name")
    sizes = [s_ for s_ in ORDER if s_ in df.index]
    x = np.arange(len(sizes)); w = 0.27

    enc = np.array([df.loc[s_, "int8_KB"] for s_ in sizes])
    tot_m = enc + np.array([df.loc[s_, "mlp8_head_KB"] for s_ in sizes])
    tot_l = enc + np.array([df.loc[s_, "lgbm_head_KB"] for s_ in sizes])

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.bar(x - w, enc, w, color="#2a78d6", zorder=3, label="encoder only")
    ax.bar(x, tot_m, w, color="#1baf7a", zorder=3, label="+ MLP-8 head")
    ax.bar(x + w, tot_l, w, color=SUP, zorder=3, label="+ LightGBM head")
    for xi, v in zip(x + w, tot_l):
        ax.text(xi, v * 1.25, f"{v:.0f}", ha="center", fontsize=6.2, color=SUP)
    for xi, v in zip(x - w, enc):
        ax.text(xi, v * 1.15, f"{v:.1f}", ha="center", fontsize=6.2, color="#2a78d6")

    ax.axhline(512, ls="--", lw=1.1, color="#8a5a06", zorder=4)
    ax.text(len(sizes) - 0.55, 300, "ESP32 SRAM budget", ha="right", va="top",
            fontsize=6.5, color="#8a5a06")
    ax.set_yscale("log")
    ax.set_ylim(0.35, 9000)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_xlabel("encoder capacity")
    ax.set_ylabel("deployed size (KB, int8)")
    ax.set_title("The classifier, not the encoder, dominates", pad=22)
    ax.legend(frameon=False, ncol=3, handlelength=1.0, columnspacing=1.0,
              handletextpad=0.4, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              fontsize=6.8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_deployment")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    print("figures:")
    for fn in (fig_srr, fig_complementary, fig_erasure, fig_deployment):
        try:
            fn()
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    print(f"-> {FIGS}")


if __name__ == "__main__":
    main()
