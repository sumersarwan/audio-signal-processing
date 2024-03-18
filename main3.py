import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def visualize_audio_features(audio_file_paths):
    def update_plot(val):
        idx = int(file_slider.val)
        audio_data, sample_rate = librosa.load(audio_file_paths[idx], sr=None)

        axs[0].cla()
        axs[0].plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data)
        axs[0].set_title(f'Waveform - {audio_file_paths[idx]}')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')

        axs[1].cla()
        axs[1].imshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max),
                      aspect='auto', origin='lower', cmap='viridis')
        axs[1].set_title(f'Spectrogram - {audio_file_paths[idx]}')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Frequency')

        axs[2].cla()
        zero_crossings = librosa.feature.zero_crossing_rate(audio_data)
        time_axis = np.linspace(0, len(zero_crossings[0]) / sample_rate, num=len(zero_crossings[0]))
        axs[2].plot(time_axis, zero_crossings[0])
        axs[2].set_title(f'Zero Crossing Rate - {audio_file_paths[idx]}')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Zero Crossing Rate')

        # Calculate the derivative of the zero-crossing rate
        zcr_derivative = np.diff(zero_crossings[0])

        # Set a threshold for anomaly detection
        anomaly_threshold = 0.02  # Adjust this threshold based on your observations

        # Identify points where the derivative exceeds the threshold (potential anomalies)
        anomaly_points = np.where(np.abs(zcr_derivative) > anomaly_threshold)[0]

        # Highlight potential anomalies on the plot
        axs[2].plot(time_axis[anomaly_points], zero_crossings[0][anomaly_points], 'ro', label='Potential Anomaly')

        axs[2].legend()
        plt.draw()

    # Rest of the code remains unchanged

    num_files = len(audio_file_paths)
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Create a slider widget for selecting the audio file
    file_slider_ax = plt.axes([0.1, 0.01, 0.8, 0.02], facecolor='lightgoldenrodyellow')
    file_slider = Slider(file_slider_ax, 'Select File', 0, num_files - 1, valinit=0, valstep=1)

    # Attach the update_plot function to the slider's on_changed event
    file_slider.on_changed(update_plot)

    plt.show()


# Replace the list with the actual paths to your audio files
audio_file_paths = ['welding-data/AE-source-1_Airpod-Cut/001.wav', 'welding-data/AE-source-1_Airpod-Cut/002.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/003.wav', 'welding-data/AE-source-1_Airpod-Cut/004.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/005.wav', 'welding-data/AE-source-1_Airpod-Cut/006.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/007.wav', 'welding-data/AE-source-1_Airpod-Cut/008.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/Background-noise.wav'
    , 'welding-data/AE-source-1_Airpod-Cut/Dry-run.wav']

visualize_audio_features(audio_file_paths)
