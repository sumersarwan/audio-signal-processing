import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def plot_waveform(ax, audio_data, sample_rate, file_name):
    ax.cla()
    ax.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data)
    ax.set_title(f'Waveform - {file_name}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')


def plot_spectrogram(ax, audio_data, file_name):
    ax.cla()
    ax.imshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max),
              aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'Spectrogram - {file_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')


def plot_zero_crossing_rate(ax, audio_data, sample_rate, file_name, anomaly_threshold=0.02):
    ax.cla()
    zero_crossings = librosa.feature.zero_crossing_rate(audio_data)
    time_axis = np.linspace(0, len(zero_crossings[0]) / sample_rate, num=len(zero_crossings[0]))
    ax.plot(time_axis, zero_crossings[0])
    ax.set_title(f'Zero Crossing Rate - {file_name}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Zero Crossing Rate')

    # Calculate the derivative of the zero-crossing rate
    zcr_derivative = np.diff(zero_crossings[0])

    # Identify points where the derivative exceeds the threshold (potential anomalies)
    anomaly_points = np.where(np.abs(zcr_derivative) > anomaly_threshold)[0]

    # Highlight potential anomalies on the plot
    ax.plot(time_axis[anomaly_points], zero_crossings[0][anomaly_points], 'ro', label='Potential Anomaly')
    ax.legend()


def update_plot(val, axs, audio_file_paths, file_slider):
    idx = int(file_slider.val)
    audio_data, sample_rate = librosa.load(audio_file_paths[idx], sr=None)

    plot_waveform(axs[0], audio_data, sample_rate, audio_file_paths[idx])
    plot_spectrogram(axs[1], audio_data, audio_file_paths[idx])
    plot_zero_crossing_rate(axs[2], audio_data, sample_rate, audio_file_paths[idx])

    plt.draw()


def visualize_audio_features(audio_file_paths):
    num_files = len(audio_file_paths)
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Create a slider widget for selecting the audio file
    file_slider_ax = plt.axes([0.1, 0.01, 0.8, 0.02], facecolor='lightgoldenrodyellow')
    file_slider = Slider(file_slider_ax, 'Select File', 0, num_files - 1, valinit=0, valstep=1)

    # Attach the update_plot function to the slider's on_changed event
    file_slider.on_changed(lambda val: update_plot(val, axs, audio_file_paths, file_slider))

    plt.show()


# Replace the list with the actual paths to your audio files
audio_file_paths = ['welding-data/AE-source-1_Airpod-Cut/001.wav', 'welding-data/AE-source-1_Airpod-Cut/002.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/003.wav', 'welding-data/AE-source-1_Airpod-Cut/004.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/005.wav', 'welding-data/AE-source-1_Airpod-Cut/006.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/007.wav', 'welding-data/AE-source-1_Airpod-Cut/008.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/Background-noise.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/Dry-run.wav']

visualize_audio_features(audio_file_paths)
