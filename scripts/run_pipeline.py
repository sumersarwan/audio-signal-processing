"""Run the end-to-end welding acoustic analytics pipeline.

Usage:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.anomaly import META_COLUMNS, save_model, score_anomalies
from src.features import build_feature_table, discover_audio_files

DATA_ROOT = ROOT / "welding-data"
OUTPUT_ROOT = ROOT / "outputs"


def plot_anomaly_timeline(scores: pd.DataFrame, output_path: Path) -> None:
    """Plot anomaly scores across time for each recording."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for file_name, group in scores.groupby("file_name", sort=True):
        ax.plot(
            group["start_sec"],
            group["anomaly_score"],
            alpha=0.65,
            label=file_name,
        )

    ax.set_title("Window-level anomaly score by recording")
    ax.set_xlabel("Window start (s)")
    ax.set_ylabel("Relative anomaly score (higher = more unusual)")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pca(features: pd.DataFrame, output_path: Path) -> None:
    """Project engineered features into two PCA dimensions."""
    feature_columns = [
        column for column in features.columns if column not in META_COLUMNS
    ]

    feature_values = features[feature_columns]
    feature_values = feature_values.fillna(feature_values.median())
    x = StandardScaler().fit_transform(feature_values)

    coords = PCA(n_components=2, random_state=42).fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 7))

    for file_name, indices in features.groupby("file_name").groups.items():
        positions = list(indices)
        ax.scatter(
            coords[positions, 0],
            coords[positions, 1],
            s=12,
            alpha=0.5,
            label=file_name,
        )

    ax.set_title("PCA projection of acoustic feature space")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    audio_files = discover_audio_files(DATA_ROOT)

    print(
        f"Found {len(audio_files)} audio files under "
        f"{DATA_ROOT.relative_to(ROOT)}"
    )

    features = build_feature_table(
        audio_files,
        sample_rate=22_050,
        window_sec=2.0,
        hop_sec=1.0,
    )

    OUTPUT_ROOT.mkdir(exist_ok=True)
    (OUTPUT_ROOT / "plots").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "models").mkdir(exist_ok=True)

    feature_path = OUTPUT_ROOT / "features.csv"
    features.to_csv(feature_path, index=False)

    scores, model = score_anomalies(features)

    score_path = OUTPUT_ROOT / "anomaly_scores.csv"
    scores.to_csv(score_path, index=False)

    save_model(
        model,
        OUTPUT_ROOT / "models" / "isolation_forest.joblib",
    )

    plot_anomaly_timeline(
        scores,
        OUTPUT_ROOT / "plots" / "anomaly_timeline.png",
    )

    plot_pca(
        features,
        OUTPUT_ROOT / "plots" / "pca_feature_space.png",
    )

    top = scores.head(10)[
        ["file_name", "start_sec", "anomaly_score", "is_anomaly"]
    ]

    print("\nTop 10 most unusual windows:")
    print(top.to_string(index=False))
    print(f"\nSaved features to: {feature_path}")
    print(f"Saved scores to:   {score_path}")


if __name__ == "__main__":
    main()
