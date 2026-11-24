"""BNCI 001-2014 MAT-file reader and deterministic EEG preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]


@dataclass
class SessionData:
    """Epoched EEG with shape (trials, channels, samples)."""

    epochs: np.ndarray
    labels: np.ndarray
    run_ids: np.ndarray
    trial_indices: np.ndarray
    artifacts: np.ndarray
    fs: float
    dropped_artifacts: int
    invalid_trials: int


def bandpass_and_car(
    signal: np.ndarray, fs: float, low_freq: float = 8.0, high_freq: float = 30.0
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filtering followed by common average reference.

    Parameters
    ----------
    signal:
        Continuous samples with shape (samples, EEG channels).
    """
    if not 0 < low_freq < high_freq < fs / 2:
        raise ValueError("Require 0 < low frequency < high frequency < Nyquist")
    sos = butter(4, [low_freq, high_freq], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, signal, axis=0)
    return filtered - filtered.mean(axis=1, keepdims=True)


def _as_vector(value: object, dtype: type) -> np.ndarray:
    return np.asarray(value, dtype=dtype).squeeze().reshape(-1)


def load_session(
    path: str | Path,
    low_freq: float = 8.0,
    high_freq: float = 30.0,
    tmin: float = 2.5,
    tmax: float = 6.0,
    drop_artifacts: bool = False,
) -> SessionData:
    """Load one training or evaluation session without mixing runs or sessions.

    In the MAT distribution, cells 1--3 contain the initial EOG calibration and
    cells 4--9 contain the six motor-imagery runs. ``trial`` is a zero-based
    sample offset when read by SciPy. The epoch is relative to fixation onset:
    the cue starts at 2 s and imagery continues until 6 s.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.download_data first.")
    if not 0 <= tmin < tmax <= 6.0:
        raise ValueError("Require 0 <= tmin < tmax <= 6 seconds")
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    runs = np.atleast_1d(content["data"])
    all_epochs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_runs: list[np.ndarray] = []
    all_trials: list[np.ndarray] = []
    all_artifacts: list[np.ndarray] = []
    dropped_artifacts = 0
    invalid_trials = 0
    sampling_rates: set[float] = set()

    for run_number, run in enumerate(runs, start=1):
        if not hasattr(run, "trial") or not hasattr(run, "y"):
            continue  # Skip the three EOG-calibration cells.
        fs = float(np.asarray(run.fs).squeeze())
        sampling_rates.add(fs)
        starts = _as_vector(run.trial, int)
        labels = _as_vector(run.y, int)
        artifacts = (
            _as_vector(run.artifacts, int)
            if hasattr(run, "artifacts")
            else np.zeros(starts.size, dtype=int)
        )
        if not (starts.size == labels.size == artifacts.size):
            raise ValueError(f"Inconsistent trial metadata in {path}, run {run_number}")
        raw = np.asarray(run.X, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < 25:
            raise ValueError(f"Expected at least 25 recorded channels in {path}")
        eeg = bandpass_and_car(raw[:, :22], fs, low_freq, high_freq)
        offset_start = int(round(tmin * fs))
        offset_stop = int(round(tmax * fs))
        selected_epochs: list[np.ndarray] = []
        selected_labels: list[int] = []
        selected_trials: list[int] = []
        selected_artifacts: list[int] = []
        for index, (start, label, artifact) in enumerate(zip(starts, labels, artifacts)):
            first = int(start) + offset_start
            last = int(start) + offset_stop
            if first < 0 or last > eeg.shape[0] or label not in (1, 2, 3, 4):
                invalid_trials += 1
                continue
            if drop_artifacts and artifact:
                dropped_artifacts += 1
                continue
            selected_epochs.append(eeg[first:last, :].T)
            selected_labels.append(int(label) - 1)
            selected_trials.append(index)
            selected_artifacts.append(int(artifact))
        if selected_epochs:
            all_epochs.append(np.stack(selected_epochs))
            all_labels.append(np.asarray(selected_labels, dtype=int))
            all_runs.append(np.full(len(selected_epochs), run_number, dtype=int))
            all_trials.append(np.asarray(selected_trials, dtype=int))
            all_artifacts.append(np.asarray(selected_artifacts, dtype=int))

    if not all_epochs:
        raise ValueError(f"No usable motor-imagery trials found in {path}")
    if sampling_rates != {250.0}:
        raise ValueError(f"Unexpected sampling rate(s) in {path}: {sampling_rates}")
    return SessionData(
        epochs=np.concatenate(all_epochs),
        labels=np.concatenate(all_labels),
        run_ids=np.concatenate(all_runs),
        trial_indices=np.concatenate(all_trials),
        artifacts=np.concatenate(all_artifacts),
        fs=sampling_rates.pop(),
        dropped_artifacts=dropped_artifacts,
        invalid_trials=invalid_trials,
    )
