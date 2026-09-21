"""Plot the rerun Workspace u20-i6 structural replay case.

The figure is generated from the extracted case summary when available;
the released script also embeds the final u20-i6 summary used in the paper. It replaces the legacy causal-effect and ordinal
severity plots with the scores and decision quantities used by the current
structural replay + OLS profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / (
    "results/rebuttal/case_studies/"
    "workspace_u20_i6_ols_gamma025_20260823/case-summary.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "user": "#4C78A8",
    "mediator": "#E45756",
    "recovery": "#59A14F",
    "live": "#F28E2B",
    "local": "#0072B2",
    "risk": "#7A5195",
    "threshold": "#CC3311",
    "alarm": "#C7B700",
    "grid": "#D8D8D8",
}


def _configure_style() -> None:
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
        serif_font = "Times New Roman"
    except ValueError:
        serif_font = "DejaVu Serif"

    plt.rcParams.update(
        {
            "font.family": serif_font,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _default_summary() -> dict[str, Any]:
    return {
        "case": {"parameters": {"gamma": 0.25}},
        "agentsentry": {
            "first_alert_boundary": 3,
            "first_alert_path": "user_preserving_mediator_projection",
            "score_sequences": {
                "S_U": [1.0, 1.0, 1.0],
                "S_M": [1.0, 1.0, 1.0],
                "S_R": [1.0, 1.0, 0.0],
                "S_L": [0.0, 0.0, 1.0],
                "sequential_risk": [0.0, 0.0, 0.0],
            },
            "boundaries": [
                {"boundary": 1, "local_gate": False, "alert": False},
                {"boundary": 2, "local_gate": False, "alert": False},
                {"boundary": 3, "local_gate": True, "alert": True},
            ],
        },
    }


def _load_summary(path: Path) -> dict[str, Any]:
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
    else:
        document = _default_summary()
    agentsentry = document["agentsentry"]
    rows = agentsentry["boundaries"]
    if not rows:
        raise ValueError(f"No boundary records in {path}")

    boundaries = np.array([int(row["boundary"]) for row in rows], dtype=int)
    expected = np.arange(1, len(rows) + 1)
    if not np.array_equal(boundaries, expected):
        raise ValueError(f"Expected contiguous boundaries, got {boundaries.tolist()}")

    score_sequences = agentsentry["score_sequences"]
    for score_name in ("S_U", "S_M", "S_R", "S_L", "sequential_risk"):
        if len(score_sequences[score_name]) != len(rows):
            raise ValueError(f"Length mismatch for {score_name}")

    first_alert = int(agentsentry["first_alert_boundary"])
    if not rows[first_alert - 1]["alert"]:
        raise ValueError("First-alert boundary is not marked as an alert")
    if agentsentry["first_alert_path"] != "user_preserving_mediator_projection":
        raise ValueError("This case is not a boundary-local projection alert")

    return document


def _decorate_axis(axis: plt.Axes, boundaries: np.ndarray) -> None:
    axis.set_xlim(0.55, boundaries[-1] + 0.45)
    axis.set_xticks(boundaries)
    axis.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.68)
    axis.set_axisbelow(True)


def _extended_step_series(
    boundaries: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.append(boundaries, boundaries[-1] + 0.34),
        np.append(values, values[-1]),
    )


def _plot_scores(axis: plt.Axes, document: dict[str, Any]) -> None:
    agentsentry = document["agentsentry"]
    sequences = agentsentry["score_sequences"]
    boundaries = np.arange(1, len(agentsentry["boundaries"]) + 1)
    alarm_boundary = int(agentsentry["first_alert_boundary"])

    specifications = [
        ("S_U", COLORS["user"], "o", "-", 3.2, 8.0, r"$S_b^{U}$ (observed-mediator user-view substitution)"),
        ("S_M", COLORS["mediator"], "s", ":", 2.8, 6.5, r"$S_b^{M}$ (probe-conditional mediator purification)"),
        ("S_R", COLORS["recovery"], "^", "-.", 2.4, 5.2, r"$S_b^{R}$ (sanitized-mediator user-view substitution)"),
        ("S_L", COLORS["live"], "D", "--", 2.2, 4.4, r"$S_b^{L}$ (user-preserving mediator purification)"),
    ]
    for key, color, marker, line_style, line_width, marker_size, label in specifications:
        values = np.asarray(sequences[key], dtype=float)
        step_boundaries, step_values = _extended_step_series(boundaries, values)
        axis.step(
            step_boundaries,
            step_values,
            where="post",
            color=color,
            linestyle=line_style,
            linewidth=line_width,
            label=label,
            zorder=3,
        )
        axis.plot(
            boundaries,
            values,
            marker=marker,
            linestyle="None",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.25,
            zorder=4,
        )

    axis.axvline(
        alarm_boundary,
        color=COLORS["alarm"],
        linestyle=(0, (2, 2)),
        linewidth=1.3,
        alpha=0.9,
        zorder=1,
    )
    axis.plot(
        alarm_boundary,
        1.0,
        marker="*",
        markersize=15,
        markerfacecolor=COLORS["alarm"],
        markeredgecolor="#776D00",
        linestyle="None",
        zorder=6,
    )
    axis.annotate(
        rf"local alarm $b^\star={alarm_boundary}$",
        xy=(alarm_boundary, 1.0),
        xytext=(alarm_boundary - 0.30, 1.18),
        textcoords="data",
        ha="right",
        fontsize=9.2,
        color="#5A5100",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#776D00"},
    )

    _decorate_axis(axis, boundaries)
    axis.set_ylim(-0.08, 1.27)
    axis.set_yticks([0, 0.5, 1.0])
    axis.set_xlabel(r"Tool-return boundary index $b$")
    axis.set_ylabel("Structural replay score")
    axis.set_title("Boundary-aligned structural replay attribution", pad=10)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(0.02, 0.42),
        ncol=2,
        frameon=True,
        framealpha=0.96,
        columnspacing=0.8,
        handlelength=2.3,
        fontsize=8.0,
    )


def _plot_decision(axis: plt.Axes, document: dict[str, Any]) -> None:
    agentsentry = document["agentsentry"]
    rows = agentsentry["boundaries"]
    boundaries = np.arange(1, len(rows) + 1)
    local_gate = np.array([float(bool(row["local_gate"])) for row in rows])
    temporal_risk = np.asarray(agentsentry["score_sequences"]["sequential_risk"], dtype=float)
    takeover = np.array([float(bool(row["alert"])) for row in rows])
    gamma = float(document["case"]["parameters"]["gamma"])
    alarm_boundary = int(agentsentry["first_alert_boundary"])

    step_boundaries, step_local_gate = _extended_step_series(boundaries, local_gate)
    axis.step(
        step_boundaries,
        step_local_gate,
        where="post",
        color=COLORS["local"],
        linewidth=3.0,
        label=r"local gate $L_b$",
        zorder=3,
    )
    axis.plot(
        boundaries,
        local_gate,
        color=COLORS["local"],
        marker="o",
        markerfacecolor=COLORS["local"],
        linestyle="None",
        markersize=5.4,
        zorder=4,
    )
    step_boundaries, step_temporal_risk = _extended_step_series(boundaries, temporal_risk)
    axis.step(
        step_boundaries,
        step_temporal_risk,
        where="post",
        color=COLORS["risk"],
        linestyle="--",
        linewidth=2.0,
        label=r"OLS risk $\mathcal{R}_b$",
        zorder=4,
    )
    axis.plot(
        boundaries,
        temporal_risk,
        color=COLORS["risk"],
        marker="s",
        markerfacecolor=COLORS["risk"],
        linestyle="None",
        markersize=5.0,
        zorder=5,
    )
    axis.axhline(
        gamma,
        color=COLORS["threshold"],
        linestyle=":",
        linewidth=1.7,
        label=rf"OLS threshold $\gamma={gamma:.2f}$",
        zorder=2,
    )
    alert_boundaries = boundaries[takeover > 0]
    axis.plot(
        alert_boundaries,
        np.full_like(alert_boundaries, 1.06, dtype=float),
        marker="*",
        markersize=15,
        markerfacecolor=COLORS["alarm"],
        markeredgecolor="#776D00",
        linestyle="None",
        label=r"takeover alarm $b^\star$",
        zorder=6,
    )
    axis.axvline(
        alarm_boundary,
        color=COLORS["alarm"],
        linestyle=(0, (2, 2)),
        linewidth=1.3,
        alpha=0.9,
        zorder=1,
    )
    axis.annotate(
        r"$L_3=1$: pending email read changes under projection",
        xy=(alarm_boundary, 1.0),
        xytext=(2.83, 1.19),
        ha="right",
        fontsize=8.8,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["local"]},
    )
    axis.annotate(
        r"$\beta_b^U=\beta_b^M=0$; OLS does not trigger",
        xy=(2.0, 0.0),
        xytext=(2.86, 0.78),
        ha="right",
        fontsize=8.8,
        color=COLORS["risk"],
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["risk"]},
    )

    _decorate_axis(axis, boundaries)
    axis.set_ylim(-0.08, 1.32)
    axis.set_yticks([0, gamma, 0.5, 1.0])
    axis.set_yticklabels(["0", f"{gamma:.2f}", "0.5", "1.0"])
    axis.set_xlabel(r"Tool-return boundary index $b$")
    axis.set_ylabel("Decision quantity")
    axis.set_title("Local gate and OLS temporal diagnostic", pad=10)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(0.035, 0.48),
        ncol=2,
        frameon=True,
        framealpha=0.96,
        columnspacing=1.1,
        handlelength=2.8,
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    return [pdf_path, png_path]


def _plot_combined(document: dict[str, Any], output_dir: Path, prefix: str) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(15.6, 4.15))
    _plot_scores(axes[0], document)
    _plot_decision(axes[1], document)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.16, wspace=0.17)
    outputs = _save_figure(figure, output_dir, f"{prefix}_combined")
    plt.close(figure)
    return outputs


def _plot_single_panels(document: dict[str, Any], output_dir: Path, prefix: str) -> list[Path]:
    outputs: list[Path] = []
    for suffix, plotter in (("scores", _plot_scores), ("decision", _plot_decision)):
        figure, axis = plt.subplots(figsize=(7.6, 3.6))
        plotter(axis, document)
        figure.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.17)
        outputs.extend(_save_figure(figure, output_dir, f"{prefix}_{suffix}"))
        plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="agentsentry_u20i6_ols_structural")
    args = parser.parse_args()

    _configure_style()
    document = _load_summary(args.summary)
    outputs = _plot_single_panels(document, args.output_dir, args.prefix)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
