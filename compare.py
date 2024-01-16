import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load_and_preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def extract_stft(audio, sr):
    stft = librosa.stft(audio)
    return stft


def plot_stft(stft, sr, title, cmap='coolwarm'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max), y_axis='log', x_axis='time', sr=sr,
                             cmap=cmap)
    plt.colorbar(format='%+2.f')
    plt.title(title)
    plt.show()


def plot_stft_difference(stft1, stft_ref, sr, title, cmap='coolwarm'):
    plt.figure(figsize=(10, 4))
    diff = np.abs(stft1 - stft_ref)
    librosa.display.specshow(librosa.amplitude_to_db(diff, ref=np.max), y_axis='log', x_axis='time', sr=sr, cmap=cmap)
    plt.colorbar(format='%+2.f')
    plt.title(title)
    plt.show()


def compare_with_reference(file_path, reference_path):
    audio, sr = load_and_preprocess_audio(file_path)
    reference_audio, _ = load_and_preprocess_audio(reference_path)

    # Ensure both audio signals have the same length
    max_length = max(len(audio), len(reference_audio))
    audio = np.pad(audio, (0, max_length - len(audio)))
    reference_audio = np.pad(reference_audio, (0, max_length - len(reference_audio)))

    stft_audio = extract_stft(audio, sr)
    stft_reference = extract_stft(reference_audio, sr)

    plot_stft(stft_audio, sr, title="STFT for Input Audio")
    # plot_stft(stft_reference, sr, title="STFT for Reference Audio")

    plot_stft_difference(stft_audio, stft_reference, sr, title="STFT Difference")


# Replace these file paths with your own audio files
file_path = "welding-data/record-007.wav"
reference_path = "welding-data/record-002.wav"

compare_with_reference(file_path, reference_path)
