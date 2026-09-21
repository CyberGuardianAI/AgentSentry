"""Plot the AgentDojo UA/ASR trade-off frontier with final AgentSentry values."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import pandas as pd

if (os.name != "nt") and os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Average UA/ASR values from the final table.
ROWS = [
    ("GPT-4o", "No Defense", 50.40, 26.17),
    ("GPT-4o", "Delimiting", 54.95, 27.27),
    ("GPT-4o", "Repeat Prompt", 65.19, 18.61),
    ("GPT-4o", "Tool Filter", 57.86, 5.58),
    ("GPT-4o", "IPIGuard", 63.25, 5.27),
    ("GPT-4o", "DeBERTa Detector", 27.21, 9.76),
    ("GPT-4o", "CausalArmor", 55.92, 12.23),
    ("GPT-4o", "AttriGuard", 72.29, 4.15),
    ("GPT-4o", "MELON", 30.25, 0.95),
    ("GPT-4o", "MELON-Aug", 45.14, 1.19),
    ("GPT-4o", "Task Shield", 53.13, 4.93),
    ("GPT-4o", "AgentSentry (ours)", 74.13, 0.00),
    ("GPT-3.5-turbo", "No Defense", 40.18, 38.93),
    ("GPT-3.5-turbo", "Delimiting", 42.00, 39.20),
    ("GPT-3.5-turbo", "Repeat Prompt", 55.33, 26.33),
    ("GPT-3.5-turbo", "Tool Filter", 60.11, 6.99),
    ("GPT-3.5-turbo", "IPIGuard", 62.11, 7.01),
    ("GPT-3.5-turbo", "DeBERTa Detector", 17.83, 14.20),
    ("GPT-3.5-turbo", "CausalArmor", 48.07, 12.95),
    ("GPT-3.5-turbo", "AttriGuard", 67.68, 1.54),
    ("GPT-3.5-turbo", "MELON", 27.79, 1.90),
    ("GPT-3.5-turbo", "MELON-Aug", 37.75, 2.14),
    ("GPT-3.5-turbo", "Task Shield", 36.68, 6.65),
    ("GPT-3.5-turbo", "AgentSentry (ours)", 71.69, 0.00),
    ("Qwen3-Max", "No Defense", 57.73, 33.22),
    ("Qwen3-Max", "Delimiting", 60.28, 28.14),
    ("Qwen3-Max", "Repeat Prompt", 60.59, 29.81),
    ("Qwen3-Max", "Tool Filter", 9.44, 0.00),
    ("Qwen3-Max", "IPIGuard", 67.47, 8.75),
    ("Qwen3-Max", "DeBERTa Detector", 30.20, 17.88),
    ("Qwen3-Max", "CausalArmor", 53.43, 12.02),
    ("Qwen3-Max", "AttriGuard", 74.89, 7.26),
    ("Qwen3-Max", "MELON", 41.10, 1.67),
    ("Qwen3-Max", "MELON-Aug", 52.77, 1.68),
    ("Qwen3-Max", "Task Shield", 50.31, 5.66),
    ("Qwen3-Max", "AgentSentry (ours)", 82.49, 0.00),
]

df = pd.DataFrame(ROWS, columns=["Model", "Defense", "UA", "ASR"])

ORDER = [
    "No Defense",
    "Delimiting",
    "Repeat Prompt",
    "Tool Filter",
    "IPIGuard",
    "DeBERTa Detector",
    "CausalArmor",
    "AttriGuard",
    "MELON",
    "MELON-Aug",
    "Task Shield",
    "AgentSentry (ours)",
]

MARKER = {
    "No Defense": "s",
    "Delimiting": "o",
    "Repeat Prompt": "^",
    "Tool Filter": "D",
    "IPIGuard": "v",
    "DeBERTa Detector": "P",
    "CausalArmor": "h",
    "AttriGuard": "p",
    "MELON": "H",
    "MELON-Aug": "8",
    "Task Shield": "*",
    "AgentSentry (ours)": "X",
}

COLOR = {
    "No Defense": "#4c78a8",
    "Delimiting": "#f58518",
    "Repeat Prompt": "#54a24b",
    "Tool Filter": "#e45756",
    "IPIGuard": "#17becf",
    "DeBERTa Detector": "#9467bd",
    "CausalArmor": "#8c564b",
    "AttriGuard": "#ff9da6",
    "MELON": "#9c755f",
    "MELON-Aug": "#e377c2",
    "Task Shield": "#7f7f7f",
    "AgentSentry (ours)": "#d8c400",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 8.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
    }
)

MODELS = ["GPT-4o", "GPT-3.5-turbo", "Qwen3-Max"]
OUT_DIR = Path(__file__).resolve().parent

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.15), sharex=True, sharey=True, dpi=300)
legend_handles = {}

for ax, model in zip(axes, MODELS):
    sub = df[df["Model"] == model].copy()
    sub["Defense"] = pd.Categorical(sub["Defense"], categories=ORDER, ordered=True)
    sub = sub.sort_values("Defense")

    for defense in ORDER:
        row = sub[sub["Defense"] == defense]
        if row.empty:
            continue

        ua = float(row["UA"].iloc[0])
        asr = float(row["ASR"].iloc[0])
        is_ours = defense == "AgentSentry (ours)"

        point = ax.scatter(
            asr,
            ua,
            s=120 if is_ours else 62,
            marker=MARKER[defense],
            color=COLOR[defense],
            edgecolor="#6f6400" if is_ours else "white",
            linewidth=1.2 if is_ours else 0.6,
            zorder=5 if is_ours else 3,
            label=defense,
        )
        legend_handles.setdefault(defense, point)

        if is_ours:
            ax.annotate(
                "AgentSentry",
                (asr, ua),
                xytext=(8, 7),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10.2,
                weight="bold",
            )

    ax.set_title(model, pad=5)
    ax.grid(True, alpha=0.25, linewidth=0.75)
    ax.set_xlim(-1.5, 42)
    ax.set_ylim(0, 90)
    ax.tick_params(axis="both", pad=1)

fig.supxlabel("Attack Success Rate (ASR) (%)", y=0.035)
fig.supylabel("Utility under Attack (UA) (%)", x=0.015)

handles = [legend_handles[name] for name in ORDER if name in legend_handles]
labels = [name for name in ORDER if name in legend_handles]
fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.885, 0.5),
    frameon=False,
    borderaxespad=0.0,
    handletextpad=0.45,
    labelspacing=0.5,
)

fig.subplots_adjust(left=0.075, right=0.865, bottom=0.23, top=0.86, wspace=0.12)
fig.savefig(OUT_DIR / "tradeoff_3panel_agentsentry_final.png", bbox_inches="tight")
fig.savefig(OUT_DIR / "tradeoff_3panel_agentsentry_final.pdf", bbox_inches="tight")

print("Saved final AgentSentry trade-off figures.")
print(df[df["Defense"] == "AgentSentry (ours)"][["Model", "UA", "ASR"]].to_string(index=False))
