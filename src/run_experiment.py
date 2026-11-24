"""Run the leakage-safe, subject-specific, cross-session decoding experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import t, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .data_loader import CHANNEL_NAMES, CLASS_NAMES, load_session
from .download_data import parse_subjects
from .features import make_baseline_classifier, make_csp_classifier
from .visualization import generate_figures


def evaluate_model(model_name, model, train, test, subject: int):
    started = time.perf_counter()
    model.fit(train.epochs, train.labels)
    fit_seconds = time.perf_counter() - started
    predicted = model.predict(test.epochs)
    metrics = {
        "subject": subject,
        "model": model_name,
        "n_train": int(train.labels.size),
        "n_test": int(test.labels.size),
        "accuracy": accuracy_score(test.labels, predicted),
        "balanced_accuracy": balanced_accuracy_score(test.labels, predicted),
        "macro_f1": f1_score(test.labels, predicted, average="macro"),
        "cohen_kappa": cohen_kappa_score(test.labels, predicted),
        "fit_seconds": fit_seconds,
        "train_artifact_trials": int(np.count_nonzero(train.artifacts)),
        "test_artifact_trials": int(np.count_nonzero(test.artifacts)),
        "train_dropped_artifacts": train.dropped_artifacts,
        "test_dropped_artifacts": test.dropped_artifacts,
        "train_invalid_trials": train.invalid_trials,
        "test_invalid_trials": test.invalid_trials,
    }
    prediction_rows = pd.DataFrame(
        {
            "subject": subject,
            "model": model_name,
            "run": test.run_ids,
            "trial": test.trial_indices,
            "true_label": test.labels,
            "predicted_label": predicted,
            "true_class": np.asarray(CLASS_NAMES)[test.labels],
            "predicted_class": np.asarray(CLASS_NAMES)[predicted],
            "artifact": test.artifacts,
        }
    )
    matrix = confusion_matrix(test.labels, predicted, labels=np.arange(4))
    return metrics, prediction_rows, matrix, model


def _confidence_interval(values: np.ndarray) -> list[float]:
    if values.size < 2:
        return [float(values.mean()), float(values.mean())]
    half_width = t.ppf(0.975, df=values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)
    return [float(values.mean() - half_width), float(values.mean() + half_width)]


def summarize_models(metrics_frame: pd.DataFrame) -> dict:
    summaries = {}
    for model_name, group in metrics_frame.groupby("model", sort=False):
        values = group["accuracy"].to_numpy()
        summaries[model_name] = {
            "mean_subject_accuracy": float(values.mean()),
            "std_subject_accuracy": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median_subject_accuracy": float(np.median(values)),
            "accuracy_95ci_across_subjects": _confidence_interval(values),
            "pooled_accuracy": float(group["accuracy"].mul(group["n_test"]).sum() / group["n_test"].sum()),
            "mean_macro_f1": float(group["macro_f1"].mean()),
            "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
            "mean_cohen_kappa": float(group["cohen_kappa"].mean()),
            "total_fit_seconds": float(group["fit_seconds"].sum()),
        }
    csp = metrics_frame[metrics_frame["model"] == "csp_lda"].sort_values("subject")
    baseline = metrics_frame[metrics_frame["model"] == "logvar_lda"].sort_values("subject")
    differences = csp["accuracy"].to_numpy() - baseline["accuracy"].to_numpy()
    if np.allclose(differences, 0):
        statistic, p_value = 0.0, 1.0
    else:
        test_result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(test_result.statistic), float(test_result.pvalue)
    comparison = {
        "mean_paired_accuracy_difference": float(differences.mean()),
        "median_paired_accuracy_difference": float(np.median(differences)),
        "cohens_dz": float(differences.mean() / differences.std(ddof=1)) if differences.size > 1 and differences.std(ddof=1) > 0 else None,
        "wilcoxon_signed_rank_statistic": statistic,
        "wilcoxon_two_sided_p_value": p_value,
        "note": "Nine paired subject-level observations; descriptive inference with low statistical power.",
    }
    return {"models": summaries, "csp_minus_baseline": comparison}


def accumulate_psd(train, sums: dict, counts: np.ndarray) -> list[float] | None:
    frequencies = None
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_epochs = train.epochs[train.labels == class_index]
        if class_epochs.shape[0] == 0:
            continue
        selected = class_epochs[:, [CHANNEL_NAMES.index("C3"), CHANNEL_NAMES.index("C4")], :]
        frequencies, spectra = welch(selected, fs=train.fs, nperseg=min(500, selected.shape[-1]), noverlap=min(250, selected.shape[-1] // 2), axis=-1)
        average = spectra.mean(axis=0)
        for channel_index, channel in enumerate(["C3", "C4"]):
            if sums[channel][class_name] is None:
                sums[channel][class_name] = np.zeros_like(average[channel_index])
            sums[channel][class_name] += average[channel_index]
        counts[class_index] += 1
    return None if frequencies is None else frequencies.tolist()


def run_experiment(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        (args.output_dir / "models").mkdir(exist_ok=True)
    all_metrics = []
    all_predictions = []
    psd_sums = {channel: {name: None for name in CLASS_NAMES} for channel in ["C3", "C4"]}
    psd_counts = np.zeros(4, dtype=int)
    frequencies = None
    example_filters = None

    for subject in args.subjects:
        print(f"\n=== Subject A{subject:02d}: train=T session, test=E session ===", flush=True)
        train = load_session(args.data_dir / f"A{subject:02d}T.mat", args.low_freq, args.high_freq, args.tmin, args.tmax, args.drop_artifacts)
        test = load_session(args.data_dir / f"A{subject:02d}E.mat", args.low_freq, args.high_freq, args.tmin, args.tmax, args.drop_artifacts)
        if train.epochs.shape[2] != test.epochs.shape[2]:
            raise RuntimeError("Training and test epoch lengths differ")
        frequencies = accumulate_psd(train, psd_sums, psd_counts)
        for model_name, model in [("csp_lda", make_csp_classifier(args.n_components)), ("logvar_lda", make_baseline_classifier())]:
            metrics, predictions, matrix, fitted = evaluate_model(model_name, model, train, test, subject)
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            print(f"{model_name}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}, kappa={metrics['cohen_kappa']:.4f}", flush=True)
            if model_name == "csp_lda" and subject == args.subjects[0]:
                csp = fitted.named_steps["onevsrestcsp"]
                example_filters = {"subject": subject, "filters": csp.filters_.transpose(0, 2, 1).tolist()}
            if args.save_models:
                joblib.dump(fitted, args.output_dir / "models" / f"A{subject:02d}_{model_name}.joblib")

    metrics_frame = pd.DataFrame(all_metrics).sort_values(["subject", "model"], ascending=[True, False])
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    metrics_frame.to_csv(args.output_dir / "per_subject_metrics.csv", index=False, float_format="%.8f")
    predictions_frame.to_csv(args.output_dir / "predictions.csv", index=False)
    for channel in ["C3", "C4"]:
        for class_index, class_name in enumerate(CLASS_NAMES):
            spectrum_sum = psd_sums[channel][class_name]
            psd_sums[channel][class_name] = (
                spectrum_sum / max(psd_counts[class_index], 1)
            ).tolist() if spectrum_sum is not None else []
    summary = {
        "project": "Subject-specific four-class motor-imagery decoding",
        "dataset": "BCI Competition IV Dataset 2a / BNCI Horizon 2020 001-2014",
        "dataset_url": "https://www.bbci.de/competition/iv/",
        "execution_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": args.subjects,
        "evaluation_protocol": "Fit on each subject's T session; evaluate once on the same subject's E session. No subject pooling and no test-driven tuning.",
        "low_freq": args.low_freq,
        "high_freq": args.high_freq,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "n_components": args.n_components,
        "drop_artifacts": args.drop_artifacts,
        "chance_accuracy": 0.25,
        "class_names": CLASS_NAMES,
        "channel_names": CHANNEL_NAMES,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "pandas", "matplotlib"]},
        },
        "results": summarize_models(metrics_frame),
        "psd": {"frequencies": frequencies, **psd_sums},
    }
    if example_filters is not None:
        summary["example_csp_filters"] = example_filters
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generate_figures(args.output_dir, args.figures_dir)
    print("\n" + json.dumps(summary["results"], indent=2), flush=True)
    print(f"\nSaved results to {args.output_dir} and figures to {args.figures_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--subjects", nargs="+", type=str, default=["1-9"])
    parser.add_argument("--low-freq", type=float, default=8.0)
    parser.add_argument("--high-freq", type=float, default=30.0)
    parser.add_argument("--tmin", type=float, default=2.5)
    parser.add_argument("--tmax", type=float, default=6.0)
    parser.add_argument("--n-components", type=int, default=4, help="Filters taken from each end of each binary CSP eigenspectrum")
    parser.add_argument("--drop-artifacts", action="store_true", help="Optional: exclude trials marked in the MAT artifact field")
    parser.add_argument("--save-models", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.subjects = parse_subjects(args.subjects)
    if not args.subjects or any(subject not in range(1, 10) for subject in args.subjects):
        raise ValueError("Select at least one subject in the range 1-9")
    run_experiment(args)


if __name__ == "__main__":
    main()
