# Subject-Specific Four-Class Motor-Imagery BCI Decoding

A complete, reproducible EEG decoding pipeline for **BCI Competition IV Dataset 2a**, built as a course
design project. The pipeline classifies four classes of motor imagery (left hand, right hand, feet,
tongue) from 22 EEG channels using one-vs-rest Common Spatial Patterns (CSP) and shrinkage Linear
Discriminant Analysis (LDA), with each of the nine subjects modeled independently.

> **Course:** Biomedical Signal Processing &nbsp;·&nbsp; **Author:** Zhao Wentao &nbsp;·&nbsp;
> **Repository:** <https://github.com/S3eze/Four-Class-Motor-Imagery-EEG-Decoding-with-Subject-Specific-CSP-and-LDA>

## Results (held-out evaluation sessions)

| Model | Mean subject accuracy | 95% CI (across subjects) | Mean macro-F1 | Mean Cohen's kappa |
|---|---|---|---|---|
| CSP-LDA (ours) | **65.28%** | 52.99% – 77.57% | 0.640 | 0.537 |
| Log-variance LDA (baseline) | 62.08% | 49.82% – 74.34% | 0.609 | 0.494 |
| Chance level | 25% | — | — | — |

- CSP spatial filtering improves accuracy on 7 of 9 subjects (mean paired gain **+3.20 pp**).
- A two-sided Wilcoxon signed-rank test across the 9 subjects gives **p = 0.3008**: the gain is
  consistent but not statistically significant at the 0.05 level with such a small sample.
- Hardest class: left-hand imagery (pooled recall 55.4%); left/right and feet/tongue confusions
  dominate the error structure.

Per-subject numbers, predictions, and all summary statistics are in `results/`
(`per_subject_metrics.csv`, `predictions.csv`, `summary.json`). Figures are in `figures/`.

## Dataset

**BCI Competition IV Dataset 2a** (Brunner et al., 2008; distributed as BNCI Horizon 2020 dataset
001-2014):

- 9 healthy subjects, two sessions each (training **T** with labels, evaluation **E**)
- 4 classes of motor imagery: left hand, right hand, feet, tongue
- 6 runs × 48 trials = 288 trials per session (72 per class)
- 22 EEG channels (10-20 system) + 3 EOG, 250 Hz
- Public research use; official page: <https://www.bbci.de/competition/iv/>

Raw MAT files are **not** committed to this repository. `src/download_data.py` downloads them
automatically (with SHA-256 hashes and structural validation) from the BNCI Horizon 2020 mirror.

## Pipeline

1. **Data acquisition** — automatic download, hash, and validation of the 18 MAT files
2. **Preprocessing** — 8–30 Hz band-pass filter, common average reference, extraction of 2.5–6 s
   motor-imagery epochs from the 22 EEG channels
3. **Feature extraction** — one-vs-rest CSP (4 components per class → 32 log-variance features)
4. **Classification** — shrinkage LDA (multi-class)
5. **Evaluation** — per-subject fitting on the T session and a single evaluation on the held-out
   E session; accuracy, macro-F1, Cohen's kappa, confusion matrices, and a paired statistical test

A log-variance baseline **without** spatial filtering is trained on the same data and splits, so the
effect of CSP can be isolated.

## Repository structure

```
src/
  download_data.py    # data download + integrity checks (primary site with mirror fallback)
  data_loader.py      # MAT parsing, filtering, epoching
  features.py         # one-vs-rest CSP + shrinkage LDA estimators
  run_experiment.py   # full experiment loop, metrics, model export
  visualization.py    # all report figures
tests/                # unit tests (synthetic data, no download needed)
results/              # metrics tables, predictions, summary.json, saved models
figures/              # generated figures (accuracy, confusion matrix, PSD, CSP filters, ...)
reports/              # course design report and code documentation (DOCX)
data/raw/             # downloaded MAT files (git-ignored)
tools/                # report generation helpers
```

## Reproduction

```bash
python -m venv .venv

# Windows
.venv\Scripts\python -m pip install -r requirements.txt
# Linux / macOS
# .venv/bin/python -m pip install -r requirements.txt

# 1. Download and validate the data (~large download)
python -m src.download_data --subjects 1-9

# 2. Run the full experiment
python -m src.run_experiment --subjects 1-9 --save-models

# 3. Regenerate figures from the results
python -c "from pathlib import Path; from src.visualization import generate_figures; generate_figures(Path('results'), Path('figures'))"

# 4. Run the tests
python -m unittest discover -s tests -v
```

Everything is deterministic apart from the download order; no random seeds are involved in the
pipeline itself (the evaluation split is the official T/E session split, not a random split).

## Evaluation protocol notes

- **No subject pooling** — every subject gets an independently fitted model, as recommended by the
  competition.
- **No test-driven tuning** — the E sessions are used exactly once for final evaluation; all
  preprocessing and hyperparameter choices follow the dataset description and course material.
- Artifact-flagged trials are retained by default (set `drop_artifacts` to change this).

## Citation

If you use the dataset, please cite the dataset description:

> C. Brunner, M. Naeem, R. Leeb, B. Graimann, and G. Pfurtscheller. "BCI Competition 2008 – Graz
> data set A." Institute for Knowledge Discovery (Brain-Computer Interface Lab), Graz University
> of Technology, 2008.

## License

The code in this repository is provided for coursework. The dataset is distributed by its original
authors for research use; see the official competition page for terms.
