import pyaudio as audio
import numpy as np
import matplotlib.pyplot as plt
# pip install --upgrade --no-cache-dir gdown
# gdown 1xGUeeM-oY0pyXA0OO8_uKwT-vLAsiyZB  # refsignal.py
# gdown 1xTibH8tNbpwdSWmkziFGS24WXEYhtMRl  # wavaudioread.py
# gdown 10f3-zLIpu81jjQtr-Mr7BCtFz6u3o4N1 # recording_tool.py

from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse
from IPython.display import Audio

# from Code.subsystems.subsystem import subSystem
#
from refsignal import refsignal  # model for the EPO4 audio beacon signal

# from wavaudioread import wavaudioread
# from recording_tool import recording_tool

# Global Variables
Fs = 44100


# Find all audio devices visible to pyaudio, if print_list = True, print the list of audio_devices
def audio_devices(*, print_list: bool):
    pyaudio_handle = audio.PyAudio()

    if print_list:
        for i in range(pyaudio_handle.get_device_count()):
            device_info = pyaudio_handle.get_device_info_by_index(i)
            print(i, device_info['name'])

    return pyaudio_handle


# audio_devices(print_list=True)


# TODO: write this as a subsystem

# from subsystems.subsystemStateEnum import subSystemState

# class Localizationsubsystem(subSystem):
#
#     def __int__(self):
#
#     def start(self):
#
#     def update(self):
#
#     def stop(self):


def microphone_array(device_index, duration_recording):
    # Fs = 44100
    number_of_samples = duration_recording * Fs
    N = number_of_samples

    pyaudio_handle = audio_devices(print_list=False)
    stream = pyaudio_handle.open(input_device_index=device_index, channels=5, format=audio.paInt16, rate=Fs, input=True)

    samples = stream.read(N)

    # samples = []
    # second_tracking = 0
    # second_count = 0
    # for i in range(0, int(Fs / N * duration_recording)):
    #     data = stream.read(N)
    #     samples.append(data)
    #     second_tracking += 1
    #     if second_tracking == Fs / N:
    #         second_count += 1
    #         second_tracking = 0
    #         print(f'Time Left: {duration_recording - second_count} seconds')
    #
    # pa = audio.PyAudio()
    # stream.stop_stream()
    # stream.close()
    # pa.terminate()

    data = np.frombuffer(samples, dtype='int16')

    data_length = len(data[::5])
    data_mic_0 = data[0::5]
    data_mic_1 = data[1::5]
    data_mic_2 = data[2::5]
    data_mic_3 = data[3::5]
    data_mic_4 = data[4::5]

    sample_axis_mic_0 = np.linspace(0, data_length / Fs, data_length)
    sample_axis_mic_1 = np.linspace(0, data_length / Fs, data_length)
    sample_axis_mic_2 = np.linspace(0, data_length / Fs, data_length)
    sample_axis_mic_3 = np.linspace(0, data_length / Fs, data_length)
    sample_axis_mic_4 = np.linspace(0, data_length / Fs, data_length)

    mic_0 = sample_axis_mic_0, data_mic_0
    mic_1 = sample_axis_mic_1, data_mic_1
    mic_2 = sample_axis_mic_2, data_mic_2
    mic_3 = sample_axis_mic_3, data_mic_3
    mic_4 = sample_axis_mic_4, data_mic_4

    return mic_0, mic_1, mic_2, mic_3, mic_4


# mics = microphone_array(1, 4)
# for i in range(5):
#     np.savetxt("Recording_handclap_"+str(i)+".csv", mics[i], delimiter=",")
#     # wavfile.write("Recording_handclap_"+str(i)+".wav", Fs, mics[i][1])
#     plt.plot(mics[i][0], mics[i][1])
#     plt.show()

# Set up transmit signal
# normal values:44100, 64, 1, 8, 2, 0x92340f0faaaa4321,
def transmit_signal(Fs, Nbits, Timer0, Timer1, Timer3, code, repetition_pulses=None):
    # Create reference signal
    x, _ = refsignal(Nbits, Timer0, Timer1, Timer3, code, Fs)

    if repetition_pulses is not None:
        nrep = repetition_pulses
        xx = np.kron(np.ones(nrep), x)
        return xx
    else:
        return x


#
#
# x = transmit_signal(1, 8, 2)
# print(x)
# # wavfile.write("Recording-6.wav", Fs, x)
#
# # PlayBack
# Fs, x = wavfile.read('Recording-6.wav')
# Audio(x, autoplay=True, rate=Fs)

# Fs_TX = 44100
# Nbits = 64
# Timer0 = 1
# Timer1 = 8
# Timer3 = 2
# code = 0x92340f0faaaa4321

# Create reference signal
# x, _ = refsignal(Nbits, Timer0, Timer1, Timer3, code, Fs_TX)
# nrep = 10
# xx = np.kron(np.ones(nrep), x)

# period = 1/Fs_TX
# x = transmit_signal(44100, 64, 1, 8, 2, 0x92340f0faaaa4321, 5)
# t = np.linspace(0, len(x), len(x))
#
# plt.plot(t, x)
# # plt.xlim(0, 600)
# plt.show()


# Fs, x = wavfile.read("C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\subsystems\Recording_handclap_0.wav")
# Audio(x, autoplay=True, rate=Fs)

data = np.loadtxt("Recording_handclap_0.csv", delimiter=",")
plt.plot(data[0], data[1])
plt.show()

