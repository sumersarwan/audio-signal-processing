"""Smoke tests for the feature-extraction layer."""

import numpy as np

from src.features import extract_features, iter_audio_windows


def test_feature_extraction_returns_expected_core_features():
    sample_rate = 22_050
    time = np.arange(sample_rate, dtype=float) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    features = extract_features(audio, sample_rate)

    assert "rms_mean" in features
    assert "zcr_mean" in features
    assert "spectral_centroid_mean" in features
    assert "mfcc_1_mean" in features
    assert np.isfinite(features["rms_mean"])


def test_windowing_preserves_timing_metadata():
    sample_rate = 1000
    audio = np.zeros(5000, dtype=np.float32)

    windows = list(
        iter_audio_windows(
            audio,
            sample_rate,
            window_sec=2,
            hop_sec=1,
            source_file="example.wav",
        )
    )

    assert len(windows) == 4
    assert windows[0].start_sec == 0
    assert windows[1].start_sec == 1
    assert windows[-1].end_sec == 5
    assert all(window.source_file == "example.wav" for window in windows)
