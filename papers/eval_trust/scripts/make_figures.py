"""papers/eval_trust/scripts/make_figures.py — Generate the 3 paper figures.

All figures regenerable from data/ in this repo (no external project state).

Figures:
  fig1_max_new_contingency.{pdf,png}: paired McNemar contingency under
    max_new=300 vs max_new=768. Visualises sec 3.4 sign flip.
  fig2_t0v2_channels.{pdf,png}: T0v2 channel distribution bar chart.
    Visualises sec 4.4.
  fig3_quantization_pareto.{pdf,png}: per-tensor vs group-32 quantization
    accuracy + cancer-key marginal history. Visualises sec 5.2 / 5.3.

Data sources (all in this repo):
  data/quantization_granularity/summary.json
  data/synthetic_4algo/marginal_history_anonymized.json

Usage:
  python papers/eval_trust/scripts/make_figures.py

Outputs go to papers/eval_trust/figures/ as both .pdf and .png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT / "papers" / "eval_trust"
FIG_DIR = PAPER_DIR / "figures"
DATA_DIR = ROOT / "data"

FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save_both(fig, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png")
    print(f"  saved {stem}.{{pdf,png}}")


def fig1_contingency() -> None:
    contingency_300 = np.array([[78, 21], [41, 60]])
    contingency_768 = np.array([[108, 29], [27, 35]])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    for ax, mat, title, p_str in [
        (axes[0], contingency_300, r"max_new_tokens$=300$ (Phase 13)",
         r"$b=21,\ c=41,\ p=\mathbf{0.015}$  (winner $+10$ pp)"),
        (axes[1], contingency_768, r"max_new_tokens$=768$ (audited)",
         r"$b=29,\ c=27,\ p=\mathbf{0.89}$  (winner $-1.0$ pp)"),
    ]:
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=110, aspect="auto")
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                color = "white" if v > 60 else "black"
                weight = "bold" if (i, j) in [(0, 1), (1, 0)] else "normal"
                ax.text(j, i, str(v), ha="center", va="center",
                        color=color, fontsize=14, fontweight=weight)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Instruct correct", "Instruct wrong"])
        ax.set_yticklabels(["Winner correct", "Winner wrong"])
        ax.set_title(title)
        ax.text(0.5, -0.30, p_str, transform=ax.transAxes,
                ha="center", va="top", fontsize=10)

    fig.suptitle(
        "Figure 1: Paired McNemar contingency before and after the truncation audit.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    save_both(fig, "fig1_max_new_contingency")
    plt.close(fig)


def fig2_channels() -> None:
    counts = {
        "A_truncated": 17,
        "A_extract_v2": 1,
        "B_stepwise": 4,
        "C_token": 8,
        "Class2": 35,
    }
    n_wrong, n_total = 65, 200

    labels = list(counts.keys())
    vals = list(counts.values())
    pct_of_wrong = [v / n_wrong * 100 for v in vals]
    pct_of_total = [v / n_total * 100 for v in vals]

    first_class_color = "#2E86AB"
    class2_color = "#A23B72"
    colors = [first_class_color] * 4 + [class2_color]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(labels))
    width = 0.4

    bars1 = ax.bar(x - width / 2, pct_of_wrong, width,
                   color=colors, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, pct_of_total, width,
           color=colors, alpha=0.45, edgecolor="black", linewidth=0.5)

    for bar, v in zip(bars1, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{v}", ha="center", va="bottom", fontsize=9)

    ax.axhline(15.0, ls="--", color="gray", lw=0.8, alpha=0.7)
    ax.text(len(labels) - 0.5, 15.5, r"$\alpha = 15\%$ of total",
            fontsize=8, color="gray", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Share (%)")
    ax.set_title(
        "Figure 2: T0v2 channel distribution on the audited winner "
        "(200 GSM8K dev items)."
    )
    ax.set_ylim(0, max(pct_of_wrong) * 1.15)
    ax.grid(axis="y", ls=":", alpha=0.4)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=first_class_color, edgecolor="black", label="First-class (repairable)"),
        Patch(facecolor=class2_color, edgecolor="black", label="Class 2 (reasoning bottleneck)"),
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.7,
              label="Solid = % of wrong (n=65)"),
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.4,
              label="Faded = % of total (n=200)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8, ncol=1)

    fig.tight_layout()
    save_both(fig, "fig2_t0v2_channels")
    plt.close(fig)


def fig3_pareto() -> None:
    quant_path = DATA_DIR / "quantization_granularity" / "summary.json"
    with open(quant_path) as f:
        qd = json.load(f)

    def acc_or(d: dict, key: str, default: float = 0.0) -> float:
        return d.get(key, {}).get("acc", default)

    pt_winner = [
        acc_or(qd["per_tensor"], "winner_int4_noprotect"),
        acc_or(qd["per_tensor"], "winner_int8_noprotect"),
        acc_or(qd["per_tensor"], "winner_fp16_noprotect"),
    ]
    pt_instruct = [
        acc_or(qd["per_tensor"], "instruct_int4_noprotect"),
        acc_or(qd["per_tensor"], "instruct_int8_noprotect"),
        acc_or(qd["per_tensor"], "instruct_fp16_noprotect"),
    ]
    g32_winner = [
        max(acc_or(qd["group_32"], f"winner_int4_p{p}") for p in ("05", "15", "50")),
        max(acc_or(qd["group_32"], f"winner_int8_p{p}") for p in ("05", "15", "50")),
        acc_or(qd["group_32"], "baseline_instruct_fp16"),
    ]
    g32_instruct = [
        max(acc_or(qd["group_32"], f"instruct_int4_p{p}") for p in ("05", "15", "50")),
        max(acc_or(qd["group_32"], f"instruct_int8_p{p}") for p in ("05", "15", "50")),
        acc_or(qd["group_32"], "baseline_instruct_fp16"),
    ]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0))
    levels = ["int4", "int8", "fp16"]
    x = np.arange(len(levels))
    width = 0.35

    axL.bar(x - width / 2, [v * 100 for v in pt_winner], width / 2,
            label="Winner (per-tensor)", color="#FF6B6B",
            edgecolor="black", linewidth=0.5)
    axL.bar(x - width / 2 + width / 2, [v * 100 for v in g32_winner], width / 2,
            label="Winner (group-32)", color="#C9302C",
            edgecolor="black", linewidth=0.5)
    axL.bar(x + width / 2, [v * 100 for v in pt_instruct], width / 2,
            label="Instruct (per-tensor)", color="#4ECDC4",
            edgecolor="black", linewidth=0.5)
    axL.bar(x + width / 2 + width / 2, [v * 100 for v in g32_instruct], width / 2,
            label="Instruct (group-32)", color="#28968F",
            edgecolor="black", linewidth=0.5)

    axL.set_xticks(x)
    axL.set_xticklabels(levels)
    axL.set_ylabel("GSM8K accuracy (%)")
    axL.set_ylim(0, 90)
    axL.set_title("(a) Granularity dominates bit-width.")
    axL.legend(fontsize=8, ncol=2)
    axL.grid(axis="y", ls=":", alpha=0.4)

    history_path = DATA_DIR / "synthetic_4algo" / "marginal_history_anonymized.json"
    with open(history_path) as f:
        h = json.load(f)
    history = h.get("history_anonymized", [])
    steps = [item["step"] for item in history]
    deltas = [item["delta_pp_vs_base"] for item in history]
    verdicts = [item["verdict"] for item in history]

    keep_color = "#2E86AB"
    neutral_color = "#999999"
    reject_color = "#D62728"
    bar_colors = []
    for v in verdicts:
        if "REJECT" in v:
            bar_colors.append(reject_color)
        elif "NEUTRAL" in v:
            bar_colors.append(neutral_color)
        else:
            bar_colors.append(keep_color)

    axR.bar(steps, deltas, color=bar_colors, edgecolor="black", linewidth=0.4)
    axR.axhline(0, color="black", lw=0.6)

    cancer_idx = next((i for i, v in enumerate(verdicts) if "REJECT" in v), None)
    if cancer_idx is not None:
        cancer_step = steps[cancer_idx]
        cancer_delta = deltas[cancer_idx]
        axR.annotate(
            f"cancer key:\nrank {cancer_step}\n($-{abs(cancer_delta):.1f}$ pp)",
            xy=(cancer_step, cancer_delta),
            xytext=(cancer_step + 4, cancer_delta - 7),
            fontsize=8, color=reject_color, ha="left",
            arrowprops=dict(arrowstyle="->", color=reject_color, lw=0.8),
        )

    axR.set_xlabel("Marginal-benefit step (saliency rank)")
    axR.set_ylabel(r"$\Delta$ acc on 30-item quick set (pp)")
    axR.set_title("(b) Marginal-benefit protect-set construction.")
    axR.grid(axis="y", ls=":", alpha=0.4)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=keep_color, edgecolor="black",
              label=f"KEEP ({sum(1 for v in verdicts if 'KEEP' in v)})"),
        Patch(facecolor=neutral_color, edgecolor="black",
              label=f"NEUTRAL ({sum(1 for v in verdicts if 'NEUTRAL' in v)})"),
        Patch(facecolor=reject_color, edgecolor="black",
              label=f"REJECT ({sum(1 for v in verdicts if 'REJECT' in v)})"),
    ]
    axR.legend(handles=legend_elems, loc="upper right", fontsize=8)

    fig.suptitle("Figure 3: Quantization granularity and the cancer-key effect.",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save_both(fig, "fig3_quantization_pareto")
    plt.close(fig)


def main() -> None:
    print("Generating figures...")
    print(f"  output dir: {FIG_DIR}")
    fig1_contingency()
    fig2_channels()
    fig3_pareto()
    print("Done.")


if __name__ == "__main__":
    main()
