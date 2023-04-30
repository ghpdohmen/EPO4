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

from subsystems.subsystem import subSystem

#
# from refsignal import refsignal  # model for the EPO4 audio beacon signal

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

from subsystems.subsystemStateEnum import subSystemState

class Localizationsubsystem(subSystem):
   enabled = False

    def __int__(self):
        # if self.enabled:


    def start(self):

    def update(self):

    def stop(self):


    def microphone_array(device_index, duration_recording):
        """
        @param duration_recording:
        @return:
        """
        # Fs = 44100
        number_of_samples = duration_recording * Fs
        N = number_of_samples

        pyaudio_handle = audio_devices(print_list=False)
        stream = pyaudio_handle.open(input_device_index=device_index, channels=5, format=audio.paInt16, rate=Fs, input=True)

        samples = stream.read(N)
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

    def gold_code(polynomial_1, polynomial_2, length):
        poly1 = [int(pol_length) for pol_length in polynomial_1]
        poly2 = [int(pol_length) for pol_length in polynomial_2]

        ones_array_poly1 = [1] * len(poly1)
        ones_array_poly2 = [1] * len(poly2)

        gold = []
        for i in range(length):
            output = ones_array_poly1[0]^ones_array_poly2[0]
            gold.append(output)

            ones_array_poly1 = [output] + ones_array_poly1[:-1]
            ones_array_poly2 = ones_array_poly1[-1] ^ [coeff * ones_array_poly2[i] for i, coeff in enumerate(poly2[::-1])] + ones_array_poly2[:-1]

        return gold

    def mic_plotter(data: bool, device_index=None, duration_recording=None):
        if data is False:
            mics = microphone_array(device_index, duration_recording)
            for i in range(5):
                plt.plot(mics[i][0], mics[i][1])
                plt.show()
            return
        else:
            for i in range(5):
                data = np.loadtxt("Recording_handclap_" + str(i) + ".csv", delimiter=",")
                plt.plot(data[0], data[1])
                plt.show()
            return



    def data_saver(device_index, duration_recording):
        mics = microphone_array(device_index, duration_recording)
        for i in range(5):
            np.savetxt("Recording_handclap_"+str(i)+".csv", mics[i], delimiter=",")
            return


# # Generates a gold code of length n using LFSRs
# def gold_code(n):
#     # Define Linear-feedback shift register taps
#     taps1 = [1, 2, 5, 6]
#     taps2 = [2, 5, 7, 8]
#
#     # initiaze the taps length of 10
#     lfsr1 = [1] * 10
#     lfsr2 = [1] * 10
#
#     # generate code
#     code = []
#     for i in range(n):
#         # compute XOR of LFSRs
#         xor = lfsr1[-1] ^ lfsr2[-1]
#
#         # Append to code
#         code.append(xor)
#
#         # Update LFSRs
#         lfsr1 = [xor if j in taps else lfsr1[j - 1] for j in range(10)]
#         lfsr2 = [xor if j in taps else lfsr2[j - 1] for j in range(10)]
#
#     return code

# p1 = "111000110101"
# p2 = "010101001100"
# length = 10
#
# code = gold_code(p1, p2, length)
# print(code)
