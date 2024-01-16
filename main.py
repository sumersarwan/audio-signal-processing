import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

scale_file = "audio/scale.wav"
noise_file = "audio/noise.wav"
welding_file = "welding-data/record-002_RUAW7S4W.wav"

ipd.Audio(scale_file)
ipd.Audio(noise_file)
ipd.Audio(welding_file)

scale, sr = librosa.load(scale_file)
noise, nr = librosa.load(noise_file)
welding, sr = librosa.load(welding_file)

FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

Y_scale = np.abs(S_scale) ** 2


#  function to plot the graph using the data provide by librosa
def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")


# for plotting the scale.wav file
# plot_spectrogram(Y_scale, sr, HOP_SIZE)  # this will plot a raw graph -- see fig 1
# Y_log_scale = librosa.power_to_db(Y_scale)  # converting to log scale
# # plot_spectrogram(Y_log_scale, sr, HOP_SIZE)  # log scale in x-axis -- see fig 2
# plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")  # log scale in y-axis -- see fig 3

# #  for plotting the noise.wav file
# S_noise = librosa.stft(noise, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
# Y_noise = librosa.power_to_db(np.abs(S_noise) ** 2)
# plot_spectrogram(Y_noise, sr, HOP_SIZE, y_axis="log")

S_noise = librosa.stft(welding, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_noise = librosa.power_to_db(np.abs(S_noise) ** 2)
plot_spectrogram(Y_noise, sr, HOP_SIZE, y_axis="log")

#  to view the graph
plt.show()
