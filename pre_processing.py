import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os


def conv_files(folder):
    for root, dirs, files in os.walk(path, topdown=False):
        for m in files:
            name, genre = m.split('-')
            spectrogram = spec(m)
            save_file(spectrogram, name, genre)

def spec(file_path):
    sample_rate, samples = wavfile.read(file_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    return spectrogram

def save_file(processed_file, name, genre):
    p1 = plt.plot(processed_file)
    plt.savefig(name + '-' + genre)


if __name__ == "__main__":
    path = "C\\Music\\project_files"
    conv_files(path)
