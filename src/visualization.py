"""Generate all figures used in the report from saved experimental outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from .data_loader import CHANNEL_NAMES, CLASS_NAMES

COLORS = {"csp_lda": "#246BCE", "logvar_lda": "#E17C3F"}


def save_pipeline_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    labels = ["Public MAT files\n9 subjects", "8-30 Hz filter\nCAR + epochs", "OvR-CSP\n32 log features", "Shrinkage LDA\n4 classes", "Held-out session\nmetrics + figures"]
    for index, label in enumerate(labels):
        x = 0.15 + index * 2.4
        rect = plt.Rectangle((x, 0.65), 1.9, 1.15, facecolor="#EAF2FD", edgecolor="#246BCE", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.95, 1.225, label, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            ax.annotate("", xy=(x + 2.32, 1.225), xytext=(x + 1.93, 1.225), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#555555"})
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_trial_figure(output: Path, tmin: float, tmax: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.3))
    ax.set_xlim(-0.3, 8.2)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    ax.hlines(0.42, 0, 7.5, color="#555555", linewidth=1.5)
    for time, label in [(0, "Fixation + tone"), (2, "Cue"), (3.25, "Cue off"), (6, "MI end"), (7.5, "Break")]:
        ax.vlines(time, 0.35, 0.52, color="#555555")
        ax.text(time, 0.18, f"{time:g} s", ha="center", fontsize=9)
        ax.text(time, 0.65, label, ha="center", fontsize=9, rotation=20 if time == 3.25 else 0)
    ax.axvspan(tmin, tmax, ymin=0.29, ymax=0.48, color="#79AEF2", alpha=0.75)
    ax.text((tmin + tmax) / 2, 0.97, f"Analysis window: {tmin:g}-{tmax:g} s", ha="center", color="#174A8B", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_figure(metrics: pd.DataFrame, output: Path) -> None:
    pivot = metrics.pivot(index="subject", columns="model", values="accuracy")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(pivot))
    width = 0.35
    for offset, model in [(-width / 2, "csp_lda"), (width / 2, "logvar_lda")]:
        values = pivot[model].to_numpy()
        bars = ax.bar(x + offset, values * 100, width, label=model.replace("_", " + ").upper(), color=COLORS[model])
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.axhline(25, linestyle="--", color="#666666", linewidth=1.2, label="Chance (25%)")
    ax.set_xticks(x, [f"A{int(s):02d}" for s in pivot.index])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_xlabel("Subject-specific model")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_confusion_figure(predictions: pd.DataFrame, output: Path) -> None:
    subset = predictions[predictions["model"] == "csp_lda"]
    matrix = np.zeros((4, 4), dtype=int)
    for row in subset.itertuples():
        matrix[int(row.true_label), int(row.predicted_label)] += 1
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, values, title, fmt in [(axes[0], matrix, "Trial counts", "d"), (axes[1], normalized, "Row-normalized", ".2f")]:
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=matrix.max() if fmt == "d" else 1)
        ax.set_xticks(range(4), ["Left", "Right", "Feet", "Tongue"], rotation=35, ha="right")
        ax.set_yticks(range(4), ["Left", "Right", "Feet", "Tongue"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                value = values[i, j]
                ax.text(j, i, format(value, fmt), ha="center", va="center", color="white" if value > (matrix.max() * 0.55 if fmt == "d" else 0.55) else "black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Pooled held-out predictions from subject-specific CSP-LDA models")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_psd_figure(psd_payload: dict, output: Path) -> None:
    frequencies = np.asarray(psd_payload["frequencies"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, channel in zip(axes, ["C3", "C4"]):
        for class_index, class_name in enumerate(CLASS_NAMES):
            spectrum = np.asarray(psd_payload[channel][class_name])
            ax.plot(frequencies, 10 * np.log10(np.maximum(spectrum, 1e-20)), label=class_name.replace("_", " ").title(), linewidth=1.5)
        ax.axvspan(8, 13, color="#B9D7FA", alpha=0.35, label="Mu" if channel == "C3" else None)
        ax.axvspan(13, 30, color="#F9D7B5", alpha=0.25, label="Beta" if channel == "C3" else None)
        ax.set_xlim(8, 30)
        ax.set_ylim(-30, 10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"Motor-cortex channel {channel}")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Power spectral density (dB/Hz)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle("Class-average training-session PSD during the analysis window")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _channel_coordinates() -> np.ndarray:
    coordinates = [
        (0, 3.5), (-2, 2.5), (-0.7, 2.5), (0, 2.5), (0.7, 2.5), (2, 2.5),
        (-3, 1), (-2, 1), (-0.7, 1), (0, 1), (0.7, 1), (2, 1), (3, 1),
        (-2, -0.5), (-0.7, -0.5), (0, -0.5), (0.7, -0.5), (2, -0.5),
        (-0.7, -2), (0, -2), (0.7, -2), (0, -3),
    ]
    return np.asarray(coordinates)


def save_csp_filters_figure(filters: np.ndarray, output: Path, subject: int) -> None:
    coordinates = _channel_coordinates()
    triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
    fig, axes = plt.subplots(4, 4, figsize=(13, 11))
    for class_index in range(4):
        for row in range(4):
            # The final four columns maximize target/rest variance. Show them
            # in descending eigenvalue order and omit ill-conditioned nullspace
            # filters introduced by common average referencing.
            component = filters.shape[2] - 1 - row
            ax = axes[row, class_index]
            values = filters[class_index, :, component]
            limit = np.max(np.abs(values)) or 1
            ax.tricontourf(triangulation, values, levels=20, cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.scatter(coordinates[:, 0], coordinates[:, 1], s=9, c="black")
            for (x, y), name in zip(coordinates, CHANNEL_NAMES):
                ax.text(x, y + 0.16, name, fontsize=6, ha="center")
            ax.set_aspect("equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(CLASS_NAMES[class_index].replace("_", " ").title(), fontsize=11)
            if class_index == 0:
                ax.text(-3.8, 1, f"Filter {row + 1}", fontsize=9, va="center", rotation=90)
    fig.suptitle(f"Class-maximizing OvR-CSP spatial filters (subject A{subject:02d}; sign is arbitrary)")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def generate_figures(results_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(results_dir / "per_subject_metrics.csv")
    predictions = pd.read_csv(results_dir / "predictions.csv")
    metadata = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    save_pipeline_figure(figures_dir / "pipeline.png")
    save_trial_figure(figures_dir / "trial_timing.png", metadata["tmin"], metadata["tmax"])
    save_accuracy_figure(metrics, figures_dir / "subject_accuracy.png")
    save_confusion_figure(predictions, figures_dir / "confusion_matrix.png")
    save_psd_figure(metadata["psd"], figures_dir / "class_psd.png")
    if "example_csp_filters" in metadata:
        filters = np.asarray(metadata["example_csp_filters"]["filters"])
        save_csp_filters_figure(filters, figures_dir / "csp_filters.png", metadata["example_csp_filters"]["subject"])
