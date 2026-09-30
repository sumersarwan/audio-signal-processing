"""Compare an input recording with a reference recording.

Usage:
    python scripts/compare_reference.py \
        welding-data/record-007.wav \
        welding-data/record-002.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_distances

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import extract_features, load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an input recording with a reference recording."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("reference_file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "plots" / "reference_comparison.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_audio, input_sr = load_audio(args.input_file)
    reference_audio, reference_sr = load_audio(args.reference_file)

    if input_sr != reference_sr:
        raise ValueError(
            "Input and reference sampling rates must match after loading."
        )

    input_features = extract_features(input_audio, input_sr)
    reference_features = extract_features(
        reference_audio,
        reference_sr,
    )

    common_features = sorted(
        set(input_features) & set(reference_features)
    )

    feature_matrix = np.asarray(
        [
            [input_features[name] for name in common_features],
            [reference_features[name] for name in common_features],
        ],
        dtype=float,
    )

    feature_matrix = SimpleImputer(
        strategy="median"
    ).fit_transform(feature_matrix)

    distance = float(
        cosine_distances(
            feature_matrix[:1],
            feature_matrix[1:],
        )[0, 0]
    )

    input_stft = librosa.stft(input_audio)
    reference_stft = librosa.stft(reference_audio)

    input_db = librosa.amplitude_to_db(
        np.abs(input_stft),
        ref=np.max,
    )
    reference_db = librosa.amplitude_to_db(
        np.abs(reference_stft),
        ref=np.max,
    )

    magnitude_diff = np.abs(
        np.abs(input_stft) - np.abs(reference_stft)
    )

    figure, axes = plt.subplots(3, 1, figsize=(12, 12))

    librosa.display.specshow(
        input_db,
        sr=input_sr,
        x_axis="time",
        y_axis="log",
        ax=axes[0],
    )
    axes[0].set_title(
        f"Input spectrogram — {args.input_file.name}"
    )

    librosa.display.specshow(
        reference_db,
        sr=reference_sr,
        x_axis="time",
        y_axis="log",
        ax=axes[1],
    )
    axes[1].set_title(
        f"Reference spectrogram — {args.reference_file.name}"
    )

    librosa.display.specshow(
        librosa.amplitude_to_db(magnitude_diff, ref=np.max),
        sr=input_sr,
        x_axis="time",
        y_axis="log",
        ax=axes[2],
    )
    axes[2].set_title(
        f"Spectrogram magnitude difference | feature cosine distance = {distance:.3f}"
    )

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    print(f"Feature cosine distance: {distance:.4f}")
    print(f"Saved comparison plot: {args.output}")


if __name__ == "__main__":
    main()
