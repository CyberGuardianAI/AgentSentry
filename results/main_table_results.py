"""Main AgentDojo result table for the released AgentSentry artifact.

The values are percentages from the paper's main aggregate defense table. The
script prints the table as Markdown by default and can also export CSV or JSON:

    python results/main_table_results.py --format markdown
    python results/main_table_results.py --format csv --output results/main_table_results.csv
    python results/main_table_results.py --format json --output results/main_table_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


COLUMNS = [
    "Model",
    "Defense",
    "NoAttack_CU",
    "NoAttack_FPR",
    "ImportantInstr_UA",
    "ImportantInstr_ASR",
    "ToolKnowledge_UA",
    "ToolKnowledge_ASR",
    "InjecAgent_UA",
    "InjecAgent_ASR",
    "Avg_UA",
    "Avg_ASR",
]


ROWS = [
    ["GPT-4o", "No Defense", 78.35, 0.00, 36.84, 33.23, 42.14, 30.00, 72.22, 15.28, 50.40, 26.17],
    ["GPT-4o", "Delimiting", 72.16, 0.00, 39.85, 48.57, 50.00, 21.43, 75.00, 11.81, 54.95, 27.27],
    ["GPT-4o", "Repeat Prompt", 84.54, 0.00, 61.35, 28.42, 57.14, 11.43, 77.08, 15.97, 65.19, 18.61],
    ["GPT-4o", "Tool Filter", 72.16, 17.53, 50.38, 5.41, 60.71, 7.86, 62.50, 3.47, 57.86, 5.58],
    ["GPT-4o", "IPIGuard", 71.13, 44.33, 61.50, 3.16, 66.43, 5.71, 61.81, 6.94, 63.25, 5.27],
    ["GPT-4o", "DeBERTa Detector", 41.24, 64.95, 18.95, 14.29, 31.43, 15.00, 31.25, 0.00, 27.21, 9.76],
    ["GPT-4o", "CausalArmor", 71.13, 57.73, 37.89, 9.17, 50.00, 15.71, 79.86, 11.81, 55.92, 12.23],
    ["GPT-4o", "AttriGuard", 73.20, 25.77, 72.03, 3.16, 64.29, 9.29, 80.56, 0.00, 72.29, 4.15],
    ["GPT-4o", "MELON", 56.70, 23.71, 18.05, 0.00, 17.86, 2.86, 54.86, 0.00, 30.25, 0.95],
    ["GPT-4o", "MELON-Aug", 69.07, 28.87, 33.68, 0.00, 37.14, 3.57, 64.58, 0.00, 45.14, 1.19],
    ["GPT-4o", "Task Shield", 71.13, 22.68, 54.14, 4.21, 39.29, 5.71, 65.97, 4.86, 53.13, 4.93],
    ["GPT-4o", "AgentSentry (ours)", 76.29, 2.06, 76.84, 0.00, 64.29, 0.00, 81.25, 0.00, 74.13, 0.00],
    ["GPT-3.5-turbo", "No Defense", 72.16, 0.00, 37.44, 29.32, 16.43, 73.57, 66.67, 13.89, 40.18, 38.93],
    ["GPT-3.5-turbo", "Delimiting", 70.10, 0.00, 38.65, 35.04, 19.29, 72.14, 68.06, 10.42, 42.00, 39.20],
    ["GPT-3.5-turbo", "Repeat Prompt", 77.32, 0.00, 58.05, 16.54, 36.43, 47.86, 71.53, 14.58, 55.33, 26.33],
    ["GPT-3.5-turbo", "Tool Filter", 73.20, 15.46, 57.74, 3.16, 63.57, 15.71, 59.03, 2.08, 60.11, 6.99],
    ["GPT-3.5-turbo", "IPIGuard", 69.07, 37.11, 60.15, 1.20, 67.86, 14.29, 58.33, 5.56, 62.11, 7.01],
    ["GPT-3.5-turbo", "DeBERTa Detector", 36.08, 60.82, 12.78, 9.02, 15.00, 33.57, 25.69, 0.00, 17.83, 14.20],
    ["GPT-3.5-turbo", "CausalArmor", 68.04, 45.36, 39.10, 4.06, 34.29, 27.14, 70.83, 7.64, 48.07, 12.95],
    ["GPT-3.5-turbo", "AttriGuard", 71.13, 20.62, 70.08, 1.05, 61.43, 3.57, 71.53, 0.00, 67.68, 1.54],
    ["GPT-3.5-turbo", "MELON", 68.04, 14.43, 21.35, 0.00, 7.14, 5.71, 54.86, 0.00, 27.79, 1.90],
    ["GPT-3.5-turbo", "MELON-Aug", 73.20, 17.53, 35.79, 0.00, 13.57, 6.43, 63.89, 0.00, 37.75, 2.14],
    ["GPT-3.5-turbo", "Task Shield", 69.07, 10.31, 34.74, 2.26, 11.43, 12.14, 63.89, 5.56, 36.68, 6.65],
    ["GPT-3.5-turbo", "AgentSentry (ours)", 75.26, 5.15, 72.93, 0.00, 66.43, 0.00, 75.69, 0.00, 71.69, 0.00],
    ["Qwen3-Max", "No Defense", 85.57, 0.00, 56.24, 33.38, 35.00, 60.71, 81.94, 5.56, 57.73, 33.22],
    ["Qwen3-Max", "Delimiting", 85.57, 0.00, 58.80, 30.98, 43.57, 48.57, 78.47, 4.86, 60.28, 28.14],
    ["Qwen3-Max", "Repeat Prompt", 87.63, 0.00, 65.41, 28.12, 38.57, 57.14, 77.78, 4.17, 60.59, 29.81],
    ["Qwen3-Max", "Tool Filter", 6.19, 95.88, 3.31, 0.00, 0.00, 0.00, 25.00, 0.00, 9.44, 0.00],
    ["Qwen3-Max", "IPIGuard", 77.32, 39.18, 64.36, 4.96, 69.29, 16.43, 68.75, 4.86, 67.47, 8.75],
    ["Qwen3-Max", "DeBERTa Detector", 54.64, 63.92, 37.89, 12.93, 22.86, 40.71, 29.86, 0.00, 30.20, 17.88],
    ["Qwen3-Max", "CausalArmor", 69.07, 65.98, 49.62, 13.23, 33.57, 21.43, 77.08, 1.39, 53.43, 12.02],
    ["Qwen3-Max", "AttriGuard", 76.29, 23.71, 79.70, 4.66, 68.57, 16.43, 76.39, 0.69, 74.89, 7.26],
    ["Qwen3-Max", "MELON", 74.23, 14.43, 40.15, 0.00, 18.57, 5.00, 64.58, 0.00, 41.10, 1.67],
    ["Qwen3-Max", "MELON-Aug", 79.38, 16.49, 53.98, 0.75, 30.71, 4.29, 73.61, 0.00, 52.77, 1.68],
    ["Qwen3-Max", "Task Shield", 76.29, 12.37, 47.97, 6.32, 31.43, 8.57, 71.53, 2.08, 50.31, 5.66],
    ["Qwen3-Max", "AgentSentry (ours)", 80.41, 4.12, 87.82, 0.00, 72.86, 0.00, 86.81, 0.00, 82.49, 0.00],
]


def records() -> list[dict[str, str | float]]:
    return [dict(zip(COLUMNS, row, strict=True)) for row in ROWS]


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_markdown() -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join(["---"] * len(COLUMNS)) + " |"
    body = [
        "| " + " | ".join(format_value(record[column]) for column in COLUMNS) + " |"
        for record in records()
    ]
    return "\n".join([header, sep, *body])


def render_csv(path: Path | None) -> str:
    output = path.open("w", newline="", encoding="utf-8") if path else sys.stdout
    close_output = path is not None
    try:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records())
    finally:
        if close_output:
            output.close()
    return "" if path else ""


def render_json(path: Path | None) -> str:
    text = json.dumps(records(), indent=2)
    if path:
        path.write_text(text + "\n", encoding="utf-8")
        return ""
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["markdown", "csv", "json"], default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.format == "markdown":
        text = render_markdown()
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    elif args.format == "csv":
        render_csv(args.output)
    else:
        text = render_json(args.output)
        if text:
            print(text)


if __name__ == "__main__":
    main()
