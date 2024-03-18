import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import pyin


def visualize_audio_features(audio_file_paths):
    def update_plot(val):
        idx = int(file_slider.val)
        audio_data, sample_rate = librosa.load(audio_file_paths[idx], sr=None)

        axs[0, 0].cla()
        axs[0, 0].plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data)
        axs[0, 0].set_title(f'Waveform - {audio_file_paths[idx]}')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Amplitude')

        axs[0, 1].cla()
        axs[0, 1].imshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max),
                         aspect='auto', origin='lower', cmap='viridis')
        axs[0, 1].set_title(f'Spectrogram - {audio_file_paths[idx]}')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Frequency')

        axs[1, 0].cla()
        zero_crossings = librosa.feature.zero_crossing_rate(audio_data)
        time_axis = np.linspace(0, len(zero_crossings[0]) / sample_rate, num=len(zero_crossings[0]))
        axs[1, 0].plot(time_axis, zero_crossings[0])
        axs[1, 0].set_title(f'Zero Crossing Rate - {audio_file_paths[idx]}')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Zero Crossing Rate')

        axs[1, 1].cla()
        rms = librosa.feature.rms(y=audio_data)[0]
        axs[1, 1].plot(time_axis, rms)
        axs[1, 1].set_title(f'RMS Energy - {audio_file_paths[idx]}')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('RMS Energy')

        axs[2, 0].cla()
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data)
        librosa.display.specshow(spectral_contrast, x_axis='time', cmap='viridis', ax=axs[2, 0])
        axs[2, 0].set_title(f'Spectral Contrast - {audio_file_paths[idx]}')
        axs[2, 0].set_xlabel('Time')
        axs[2, 0].set_ylabel('Frequency Band')

        axs[2, 1].cla()
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time', cmap='viridis', ax=axs[2, 1])
        axs[2, 1].set_title(f'MFCC - {audio_file_paths[idx]}')
        axs[2, 1].set_xlabel('Time')
        axs[2, 1].set_ylabel('MFCC Coefficient')

        axs[3, 0].cla()
        hnr = librosa.effects.harmonic(audio_data)
        axs[3, 0].plot(np.linspace(0, len(hnr) / sample_rate, num=len(hnr)), hnr)
        axs[3, 0].set_title(f'Harmonic-to-Noise Ratio (HNR) - {audio_file_paths[idx]}')
        axs[3, 0].set_xlabel('Time (s)')
        axs[3, 0].set_ylabel('Amplitude')

        # Add modifications for additional subplots if needed

        plt.draw()

    num_files = len(audio_file_paths)
    fig, axs = plt.subplots(4, 2, figsize=(16, 24))  # Adjust the figure size as needed

    # Create a slider widget for selecting the audio file
    file_slider_ax = plt.axes([0.1, 0.01, 0.8, 0.02], facecolor='lightgoldenrodyellow')
    file_slider = Slider(file_slider_ax, 'Select File', 0, num_files - 1, valinit=0, valstep=1)

    # Attach the update_plot function to the slider's on_changed event
    file_slider.on_changed(update_plot)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


# Replace the list with the actual paths to your audio files
audio_file_paths = ['welding-data/AE-source-1_Airpod-Cut/001.wav', 'welding-data/AE-source-1_Airpod-Cut/002.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/003.wav', 'welding-data/AE-source-1_Airpod-Cut/004.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/005.wav', 'welding-data/AE-source-1_Airpod-Cut/006.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/007.wav', 'welding-data/AE-source-1_Airpod-Cut/008.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/Background-noise.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/Dry-run.wav']

visualize_audio_features(audio_file_paths)
