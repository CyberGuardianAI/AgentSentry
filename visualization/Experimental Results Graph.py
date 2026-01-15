import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------
# Force Times New Roman style (IEEE / USENIX friendly)
# ----------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
# ----------------------------
# Data (avg_UA, avg_ASR)
# ----------------------------
rows = [
("GPT-3.5-turbo","DeBERTa Detector",17.82,14.2),
("GPT-3.5-turbo","Delimiting",42,39.2),
("GPT-3.5-turbo","MELON",27.78,1.9),
("GPT-3.5-turbo","MELON-Aug",37.75,2.14),
("GPT-3.5-turbo","No Defense",40.18,38.93),
("GPT-3.5-turbo","Repeat Prompt",55.34,26.33),
("GPT-3.5-turbo","AgentSentry (ours)",70.16,0),
("GPT-3.5-turbo","Task Shield",36.59,6.65),
("GPT-3.5-turbo","Tool Filter",60.11,6.98),

("GPT-4o","DeBERTa Detector",27.21,9.76),
("GPT-4o","Delimiting",54.95,27.27),
("GPT-4o","MELON",30.2,1.2),
("GPT-4o","MELON-Aug",45.13,1.19),
("GPT-4o","No Defense",50.4,26.17),
("GPT-4o","Repeat Prompt",65.19,18.61),
("GPT-4o","AgentSentry (ours)",73.9,0),
("GPT-4o","Task Shield",53.13,4.93),
("GPT-4o","Tool Filter",57.9,5.58),

("Qwen3-Max","DeBERTa Detector",30.2,17.88),
("Qwen3-Max","Delimiting",60.28,28.14),
("Qwen3-Max","MELON",41.1,1.67),
("Qwen3-Max","MELON-Aug",52.77,1.68),
("Qwen3-Max","No Defense",57.73,33.22),
("Qwen3-Max","Repeat Prompt",60.69,52.18),
("Qwen3-Max","AgentSentry (ours)",79.58,0),
("Qwen3-Max","Task Shield",50.31,5.66),
("Qwen3-Max","Tool Filter",9.44,0),
]
df = pd.DataFrame(rows, columns=["Model","Defense","UA","ASR"])

# ----------------------------
# Ordering, markers
# ----------------------------
order = [
    "No Defense","Delimiting","Repeat Prompt","Tool Filter",
    "DeBERTa Detector","MELON","MELON-Aug","Task Shield","AgentSentry (ours)"
]
marker = {
    "No Defense": "s",
    "Delimiting": "o",
    "Repeat Prompt": "^",
    "Tool Filter": "D",
    "DeBERTa Detector": "P",
    "MELON": "h",
    "MELON-Aug": "p",
    "Task Shield": "*",
    "AgentSentry (ours)": "X",
}

# ----------------------------
# Global aesthetics (more compact)
# ----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.3,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
})

models = ["GPT-4o", "GPT-3.5-turbo", "Qwen3-Max"]

# Shared axis range
x_min, x_max = 0, 85
y_min, y_max = 0, 55

# ----------------------------
# Figure: tighter panels + reserve space for right-side legend
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.9), sharex=True, sharey=True)

legend_handles = {}

for ax, model in zip(axes, models):
    sub = df[df["Model"] == model].copy()
    sub["Defense"] = pd.Categorical(sub["Defense"], categories=order, ordered=True)
    sub = sub.sort_values("Defense")

    for d in order:
        r = sub[sub["Defense"] == d]
        if r.empty:
            continue

        ua = float(r["UA"].iloc[0])
        asr = float(r["ASR"].iloc[0])

        s = 95 if d == "AgentSentry (ours)" else 72   # smaller markers for density
        lw = 0.9

        sc = ax.scatter(
            ua, asr,
            s=s,
            marker=marker[d],
            linewidths=lw,
            zorder=3,
            label=d
        )

        if d not in legend_handles:
            legend_handles[d] = sc

        if d == "AgentSentry (ours)":
            ax.annotate(
                "AgentSentry",
                (ua, asr),
                textcoords="offset points",
                xytext=(-14, 6),
                ha="left",
                va="bottom",
                fontsize=11
            )

    ax.set_title(model, pad=4)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Make tick labels a bit tighter to the axes
    ax.tick_params(axis="both", pad=1)

# ----------------------------
# Shared labels: bring x-label closer to axes (less gap)
# ----------------------------
fig.supxlabel("Utility under Attack (UA) (%)", y=0.03)   # <-- closer to plots
fig.supylabel("ASR (%)", x=0.015)

# ----------------------------
# Legend: move to the right (vertical), so the whole figure is denser
# ----------------------------
handles = [legend_handles[d] for d in order if d in legend_handles]
labels = [d for d in order if d in legend_handles]

fig.legend(
    handles, labels,
    loc="center left",
    bbox_to_anchor=(0.88, 0.5),   # right side
    frameon=False,
    borderaxespad=0.0,
    handletextpad=0.5,
    labelspacing=0.6
)

# ----------------------------
# Tighten layout: very small margins; reserve right strip for legend
# ----------------------------
fig.subplots_adjust(
    left=0.075, right=0.84, bottom=0.22, top=0.86,
    wspace=0.12, hspace=0.0
)

# Save
fig.savefig("tradeoff_3panel_dense.png", dpi=500, bbox_inches="tight")
fig.savefig("tradeoff_3panel_dense.pdf", bbox_inches="tight")
plt.show()
