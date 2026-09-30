"""Reusable audio loading, segmentation and feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


@dataclass(frozen=True)
class AudioWindow:
    """A fixed-duration audio segment with source timing metadata."""

    source_file: str
    start_sec: float
    end_sec: float
    audio: np.ndarray
    sample_rate: int


def load_audio(path: str | Path, sample_rate: int = 22_050) -> tuple[np.ndarray, int]:
    """Load a supported audio file as mono at a consistent sample rate."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    audio, sr = librosa.load(path, sr=sample_rate, mono=True)

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    return audio.astype(np.float32), sr


def iter_audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    source_file: str = "",
    drop_last: bool = True,
) -> Iterable[AudioWindow]:
    """Yield overlapping fixed-duration windows."""
    if window_sec <= 0 or hop_sec <= 0:
        raise ValueError("window_sec and hop_sec must be positive.")

    window = max(1, int(window_sec * sample_rate))
    hop = max(1, int(hop_sec * sample_rate))

    for start in range(0, len(audio), hop):
        stop = start + window

        if stop > len(audio):
            if drop_last:
                break
            chunk = np.pad(audio[start:], (0, stop - len(audio)))
        else:
            chunk = audio[start:stop]

        if len(chunk) < window:
            continue

        yield AudioWindow(
            source_file=source_file,
            start_sec=start / sample_rate,
            end_sec=stop / sample_rate,
            audio=chunk,
            sample_rate=sample_rate,
        )


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return descriptive statistics for a one-dimensional feature series."""
    values = np.asarray(values, dtype=float)

    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_median": float(np.nanmedian(values)),
    }


def extract_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Extract interpretable time-, spectral- and perceptual-domain features."""
    if len(audio) < 32:
        raise ValueError("Audio segment is too short for feature extraction.")

    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(len(audio))))))
    hop_length = max(64, n_fft // 4)

    features: dict[str, float] = {}

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        hop_length=hop_length,
    )[0]

    rms = librosa.feature.rms(
        y=audio,
        frame_length=n_fft,
        hop_length=hop_length,
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    features.update(_summary(zcr, "zcr"))
    features.update(_summary(rms, "rms"))
    features.update(_summary(centroid, "spectral_centroid"))
    features.update(_summary(bandwidth, "spectral_bandwidth"))
    features.update(_summary(rolloff, "spectral_rolloff"))
    features.update(_summary(flatness, "spectral_flatness"))

    for i, row in enumerate(contrast, start=1):
        features.update(_summary(row, f"spectral_contrast_{i}"))

    for i, row in enumerate(mfcc, start=1):
        features.update(_summary(row, f"mfcc_{i}"))

    features["peak_amplitude"] = float(np.max(np.abs(audio)))
    features["signal_std"] = float(np.std(audio))

    return features


def build_feature_table(
    audio_paths: Iterable[str | Path],
    sample_rate: int = 22_050,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
) -> pd.DataFrame:
    """Create a window-level feature table and retain recording identity."""
    rows: list[dict[str, float | str]] = []

    for path in audio_paths:
        path = Path(path)
        audio, sr = load_audio(path, sample_rate=sample_rate)

        for window in iter_audio_windows(
            audio,
            sr,
            window_sec=window_sec,
            hop_sec=hop_sec,
            source_file=str(path),
        ):
            row: dict[str, float | str] = {
                "source_file": str(path),
                "file_name": path.name,
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "sample_rate": float(sr),
            }
            row.update(extract_features(window.audio, sr))
            rows.append(row)

    if not rows:
        raise ValueError("No valid audio windows were extracted.")

    return pd.DataFrame(rows)


def discover_audio_files(root: str | Path) -> list[Path]:
    """Recursively discover supported audio files under a root directory."""
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
