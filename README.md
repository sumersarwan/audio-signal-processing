# Acoustic Signal Analytics for Welding Process Monitoring

An end-to-end analytics project that applies **digital signal processing, feature engineering and unsupervised machine learning** to acoustic signals generated during welding experiments.

The project explores whether acoustic signatures can be transformed into measurable features that support **process monitoring, anomaly detection and condition-based investigation**.

> **Project positioning:** manufacturing domain problem + signal processing + data analytics + machine learning.

---

## Problem Statement

Welding generates acoustic signals that vary across time and frequency. Raw recordings are difficult to inspect directly, so this project converts them into structured signal representations and machine-learning-ready features.

### Core Question

**Can acoustic signals be transformed into reliable features that help identify unusual welding-process behaviour?**

This repository is an exploratory analytics prototype. The current data does **not** provide a sufficiently documented set of process-condition labels for a defensible supervised classification benchmark, so the main ML workflow uses **unsupervised anomaly detection**.

---

## End-to-End Pipeline

```text
Raw Welding Audio
        |
        v
Audio Loading & Standardisation
        |
        v
2-second windows / 1-second hop
        |
        +---------------------------+
        |                           |
        v                           v
Time-domain analysis         Frequency-domain analysis
        |                           |
        +-------------+-------------+
                      |
                      v
              Feature Engineering
                      |
                      v
             Feature Matrix (CSV)
                      |
                      v
             Isolation Forest
                      |
                      v
             Relative Anomaly Score
                      |
                      v
        Visualisation & Investigation
```

---

## Signal Processing

The original analysis explored:

- Waveform visualisation
- Short-Time Fourier Transform (STFT)
- Spectrograms
- Log-scaled frequency representations
- Zero-Crossing Rate (ZCR)
- RMS energy
- Spectral analysis
- Signal-to-reference comparison

Selected historical visual outputs are retained under [results/](results/) for reference.

---

## Feature Engineering

Each audio window is converted into interpretable numerical features.

| Feature family | Examples |
| --- | --- |
| Time-domain | RMS energy, Zero-Crossing Rate, signal standard deviation, peak amplitude |
| Spectral | Spectral centroid, bandwidth, rolloff, flatness |
| Spectral shape | Spectral contrast |
| Perceptual | 13 MFCC coefficients |
| Aggregation | Mean, standard deviation and median per feature |

The final feature table also retains the **source recording and window start/end time**, enabling analysis at the recording level and helping prevent accidental leakage when future supervised models are added.

---

## Unsupervised Machine Learning

The current model uses **Isolation Forest** to identify windows whose engineered acoustic signatures differ from the overall population.

The output contains:

- `anomaly_score`: relative unusualness; higher values are more unusual
- `is_anomaly`: Isolation Forest flag based on the model's internal decision boundary

The score is **relative, not a calibrated probability of failure**. Any operational threshold would need to be established using labeled process data and domain validation.

---

## Reference Comparison

`scripts/compare_reference.py` provides a second analytical view:

1. Load an input recording and a reference recording.
2. Extract the same engineered feature set.
3. Calculate a feature-space cosine distance.
4. Compare normalised frequency spectra.
5. Generate a three-panel spectrogram comparison.

Example:

```bash
python scripts/compare_reference.py \
    welding-data/record-007.wav \
    welding-data/record-002.wav
```

---

## Repository Structure

```text
audio-signal-processing/
|
+-- audio/                         # small teaching/demo audio files
|
+-- welding-data/                  # experimental acoustic recordings
|
+-- notebooks/
|   +-- 04_machine_learning.ipynb # portfolio-friendly walkthrough
|
+-- src/
|   +-- __init__.py
|   +-- features.py                # loading, windowing, feature extraction
|   +-- anomaly.py                 # Isolation Forest pipeline
|
+-- scripts/
|   +-- run_pipeline.py            # end-to-end feature + anomaly pipeline
|   +-- compare_reference.py       # reference comparison
|
+-- tests/
|   +-- test_features.py           # lightweight smoke tests
|
+-- results/                       # historical visual outputs
|
+-- outputs/                       # generated analytics (created at runtime)
|
+-- requirements.txt
+-- .gitignore
+-- README.md
```

---

## Getting Started

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python scripts/run_pipeline.py
```

The pipeline discovers supported audio files recursively under `welding-data/` and writes:

```text
outputs/
+-- features.csv
+-- anomaly_scores.csv
+-- plots/
|   +-- anomaly_timeline.png
|   +-- pca_feature_space.png
+-- models/
    +-- isolation_forest.joblib
```

Generated CSV/model files are ignored by Git so the repository stays lightweight and reproducible.

### 4. Run smoke tests

```bash
pytest
```

---

## Visual Outputs

The repository retains the original spectrogram, comparison and zero-crossing outputs under `results/`. The reproducible pipeline additionally generates `outputs/plots/anomaly_timeline.png` and `outputs/plots/pca_feature_space.png` when run locally.

---

## Technical Notes

### Why window-level features?

A long recording can contain multiple process events. Fixed windows let the analysis locate unusual behaviour in time rather than assigning one score to an entire recording.

### Why keep recording identity?

Adjacent windows from one recording are highly related. If supervised learning is introduced later, windows from the same source recording should be kept within the same train/validation/test group to reduce data leakage.

### Why not claim classification accuracy?

The current repository does not contain sufficiently documented process-condition labels. Reporting classification accuracy without defensible labels and grouped validation would overstate what the data supports.

---

## Potential Extensions

The project can be extended into a stronger industrial analytics prototype by adding:

- documented welding-process condition labels
- recording-level grouped cross-validation
- supervised models such as Random Forest, SVM or XGBoost
- baseline-versus-condition feature analysis
- calibrated anomaly thresholds
- real-time audio ingestion
- Mel-spectrogram CNN experiments
- shop-floor dashboard/API deployment
- integration with production, quality and maintenance records

---

## Tech Stack

**Python · NumPy · Pandas · SciPy · Librosa · Matplotlib · Scikit-learn · Joblib · Jupyter**

---

## Project Scope

This repository is intended as an **analytics prototype and portfolio project**, not as a production-quality welding inspection system. Any safety- or quality-critical use would require validated sensors, controlled experiments, labeled data and domain-specific verification.
