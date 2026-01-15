import numpy as np
from collections import OrderedDict

def mean2(xs):
    return round(float(np.mean(xs)), 2)

def is_num(x):
    return isinstance(x, (int, float))


DATA = OrderedDict({
    "GPT-4o": OrderedDict({
        "No Defense": dict(
            cite="",
            CU=78.35, FPR=0.00,
            imp_UA=36.84, imp_ASR=33.23,
            tool_UA=42.14, tool_ASR=30.00,
            inj_UA=72.22, inj_ASR=15.28,
        ),
        "Delimiting": dict(
            cite="~\\cite{chen2025struq,hines2024defending}",
            CU=72.16, FPR=0.00,
            imp_UA=39.85, imp_ASR=48.57,
            tool_UA=50.00, tool_ASR=21.43,
            inj_UA=75.00, inj_ASR=11.81,
        ),
        "Repeat Prompt": dict(
            cite="~\\cite{learnprompting2024sandwich}",
            CU=84.54, FPR=0.00,
            imp_UA=61.35, imp_ASR=28.42,
            tool_UA=57.14, tool_ASR=11.43,
            inj_UA=77.08, inj_ASR=15.97,
        ),
        "Tool Filter": dict(
            cite="~\\cite{debenedetti2024agentdojo}",
            CU=72.16, FPR=0.00,
            imp_UA=50.38, imp_ASR=5.41,
            tool_UA=60.81, tool_ASR=7.86,
            inj_UA=62.50, inj_ASR=3.47,
        ),
        "DeBERTa Detector": dict(
            cite="~\\cite{kokkula2024palisade}",
            CU=41.24, FPR=0.00,
            imp_UA=18.95, imp_ASR=14.29,
            tool_UA=31.43, tool_ASR=15.00,
            inj_UA=31.25, inj_ASR=0.00,
        ),
        "MELON": dict(
            cite="~\\cite{zhu2025melon}",
            CU=56.70, FPR=0.00,
            imp_UA=18.05, imp_ASR=0.75,
            tool_UA=17.86, tool_ASR=2.86,
            inj_UA=54.68, inj_ASR=0.00,
        ),
        "MELON-Aug": dict(
            cite="~\\cite{zhu2025melon}",
            CU=69.07, FPR=0.00,
            imp_UA=33.68, imp_ASR=0.00,
            tool_UA=37.14, tool_ASR=3.57,
            inj_UA=64.58, inj_ASR=0.00,
        ),
        "Task Shield": dict(
            cite="~\\cite{jia2024taskshield}",
            CU=71.13, FPR=0.00,
            imp_UA=54.14, imp_ASR=4.21,
            tool_UA=39.29, tool_ASR=5.71,
            inj_UA=65.97, inj_ASR=4.86,
        ),
        "agentsentry (ours)": dict(
            cite="",
            CU=78.35, FPR=0.00,
            imp_UA=75.49, imp_ASR=0.00,
            tool_UA=63.57, tool_ASR=0.00,
            inj_UA=82.64, inj_ASR=0.00,
        ),
    }),

    "GPT-3.5-turbo": OrderedDict({
        "No Defense": dict(
            cite="",
            CU=72.16, FPR=0.00,
            imp_UA=37.44, imp_ASR=29.32,
            tool_UA=16.43, tool_ASR=73.57,
            inj_UA=66.67, inj_ASR=13.89,
        ),
        "Delimiting": dict(
            cite="~\\cite{chen2025struq,hines2024defending}",
            CU=70.10, FPR=0.00,
            imp_UA=38.65, imp_ASR=35.04,
            tool_UA=19.29, tool_ASR=72.14,
            inj_UA=68.05, inj_ASR=10.42,
        ),
        "Repeat Prompt": dict(
            cite="~\\cite{learnprompting2024sandwich}",
            CU=77.32, FPR=0.00,
            imp_UA=58.05, imp_ASR=16.54,
            tool_UA=36.43, tool_ASR=47.86,
            inj_UA=71.53, inj_ASR=14.58,
        ),
        "Tool Filter": dict(
            cite="~\\cite{debenedetti2024agentdojo}",
            CU=73.20, FPR=0.00,
            imp_UA=57.74, imp_ASR=3.16,
            tool_UA=63.57, tool_ASR=15.71,
            inj_UA=59.03, inj_ASR=2.08,
        ),
        "DeBERTa Detector": dict(
            cite="~\\cite{kokkula2024palisade}",
            CU=36.08, FPR=0.00,
            imp_UA=12.78, imp_ASR=9.02,
            tool_UA=15.00, tool_ASR=33.57,
            inj_UA=25.69, inj_ASR=0.00,
        ),
        "MELON": dict(
            cite="~\\cite{zhu2025melon}",
            CU=68.04, FPR=0.00,
            imp_UA=21.35, imp_ASR=0.00,
            tool_UA=7.14, tool_ASR=5.71,
            inj_UA=54.86, inj_ASR=0.00,
        ),
        "MELON-Aug": dict(
            cite="~\\cite{zhu2025melon}",
            CU=73.20, FPR=0.00,
            imp_UA=35.79, imp_ASR=0.00,
            tool_UA=13.57, tool_ASR=6.43,
            inj_UA=63.89, inj_ASR=0.00,
        ),
        "Task Shield": dict(
            cite="~\\cite{jia2024taskshield}",
            CU=69.07, FPR=0.00,
            imp_UA=34.74, imp_ASR=2.26,
            tool_UA=11.43, tool_ASR=12.14,
            inj_UA=63.61, inj_ASR=5.56,
        ),
        "agentsentry (ours)": dict(
            cite="",
            CU=77.32, FPR=0.00,
            imp_UA=71.88, imp_ASR=0.00,
            tool_UA=65.00, tool_ASR=0.00,
            inj_UA=73.61, inj_ASR=0.00,
        ),
    }),

    "Qwen3-Max": OrderedDict({
        "No Defense": dict(
            cite="",
            CU=85.57, FPR=0.00,
            imp_UA=56.24, imp_ASR=33.38,
            tool_UA=35.00, tool_ASR=60.71,
            inj_UA=81.94, inj_ASR=5.56,
        ),
        "Delimiting": dict(
            cite="~\\cite{chen2025struq,hines2024defending}",
            CU=85.57, FPR=0.00,
            imp_UA=58.80, imp_ASR=30.98,
            tool_UA=43.57, tool_ASR=48.57,
            inj_UA=78.47, inj_ASR=4.86,
        ),
        "Repeat Prompt": dict(
            cite="~\\cite{learnprompting2024sandwich}",
            CU=87.63, FPR=0.00,
            imp_UA=65.71, imp_ASR=95.24,
            tool_UA=38.57, tool_ASR=57.14,
            inj_UA=77.78, inj_ASR=4.17,
        ),
        "Tool Filter": dict(
            cite="~\\cite{debenedetti2024agentdojo}",
            CU=6.19, FPR=0.00,
            imp_UA=3.31, imp_ASR=0.00,
            tool_UA=0.00, tool_ASR=0.00,
            inj_UA=25.00, inj_ASR=0.00,
        ),
        "DeBERTa Detector": dict(
            cite="~\\cite{kokkula2024palisade}",
            CU=54.64, FPR=0.00,
            imp_UA=37.89, imp_ASR=12.93,
            tool_UA=22.86, tool_ASR=40.71,
            inj_UA=29.86, inj_ASR=0.00,
        ),
        "MELON": dict(
            cite="~\\cite{zhu2025melon}",
            CU=74.23, FPR=0.00,
            imp_UA=40.15, imp_ASR=0.00,
            tool_UA=18.57, tool_ASR=5.00,
            inj_UA=64.58, inj_ASR=0.00,
        ),
        "MELON-Aug": dict(
            cite="~\\cite{zhu2025melon}",
            CU=79.38, FPR=0.00,
            imp_UA=53.98, imp_ASR=0.75,
            tool_UA=30.71, tool_ASR=4.29,
            inj_UA=73.61, inj_ASR=0.00,
        ),
        "Task Shield": dict(
            cite="~\\cite{jia2024taskshield}",
            CU=76.29, FPR=0.00,
            imp_UA=47.97, imp_ASR=6.32,
            tool_UA=31.43, tool_ASR=8.57,
            inj_UA=71.53, inj_ASR=2.08,
        ),
        "agentsentry (ours)": dict(
            cite="",
            CU=83.51, FPR=0.00,
            imp_UA=86.92, imp_ASR=0.00,
            tool_UA=65.71, tool_ASR=0.00,
            inj_UA=86.11, inj_ASR=0.00,
        ),
    }),
})

# 需要比较的列（每个模型块内比较最优）
# max_cols: 越大越好；min_cols: 越小越好
max_cols = ["CU", "imp_UA", "tool_UA", "inj_UA", "avg_UA"]
min_cols = ["FPR", "imp_ASR", "tool_ASR", "inj_ASR", "avg_ASR"]

def compute_avg_fields(d):
    d = d.copy()
    d["avg_UA"]  = mean2([d["imp_UA"],  d["tool_UA"],  d["inj_UA"]])
    d["avg_ASR"] = mean2([d["imp_ASR"], d["tool_ASR"], d["inj_ASR"]])
    return d

def best_values_per_model(model_rows):
    # 先补齐 avg
    rows = {k: compute_avg_fields(v) for k, v in model_rows.items()}

    best = {}
    for col in max_cols:
        best[col] = max(rows[k][col] for k in rows if is_num(rows[k][col]))
    for col in min_cols:
        best[col] = min(rows[k][col] for k in rows if is_num(rows[k][col]))
    return rows, best

def fmt(val, bold=False):
    s = f"{val:.2f}" if isinstance(val, float) else str(val)
    return f"\\textbf{{{s}}}" if bold else s

def latex_rows(model_name, model_rows, model_tag="BaselineGray"):
    rows, best = best_values_per_model(model_rows)
    out = []

    first = True
    for defense, v in rows.items():
        cite = model_rows[defense].get("cite", "")
        name_cell = f"\\textbf{{{model_name}}}" if first else ""
        first = False

        def b(col):
            return (v[col] == best[col])

        line = (
            f"\\rowcolor{{{model_tag}}}\n"
            f"\\cellcolor{{white}}{name_cell}\n"
            f"& {defense}{cite}\n"
            f"  & {fmt(v['CU'], b('CU'))} & {fmt(v['FPR'], b('FPR'))}\n"
            f"  & {fmt(v['imp_UA'], b('imp_UA'))} & {fmt(v['imp_ASR'], b('imp_ASR'))}\n"
            f"  & {fmt(v['tool_UA'], b('tool_UA'))} & {fmt(v['tool_ASR'], b('tool_ASR'))}\n"
            f"  & {fmt(v['inj_UA'], b('inj_UA'))} & {fmt(v['inj_ASR'], b('inj_ASR'))}\n"
            f"  & {fmt(v['avg_UA'], b('avg_UA'))} & {fmt(v['avg_ASR'], b('avg_ASR'))} \\\\"
        )
        out.append(line)

    return "\n\n".join(out)

for model in DATA:
    print("="*20, model, "="*20)
    print(latex_rows(model, DATA[model], model_tag="BaselineGray"))

