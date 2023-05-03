import pyaudio as audio
import numpy as np
import matplotlib.pyplot as plt
import random

import scipy
# pip install --upgrade --no-cache-dir gdown
# gdown 1xGUeeM-oY0pyXA0OO8_uKwT-vLAsiyZB  # refsignal.py
# gdown 1xTibH8tNbpwdSWmkziFGS24WXEYhtMRl  # wavaudioread.py
# gdown 10f3-zLIpu81jjQtr-Mr7BCtFz6u3o4N1 # recording_tool.py

from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse
from IPython.display import Audio

# from subsystems.subsystem import subSystem
#
# from Code.robot import robot

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
    #    enabled = False
    #
    # def __int__(self):
    #     robot.speakerOn = False
    #     robot.code = "A23"
    #     robot.carrierFrequency = 6000
    #     robot.bitFrequency = 2000
    #     robot.repetitionCount = 64
    #     robot.speakerOn = True


#
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


# def gold_code(polynomial_1, polynomial_2, length):
#     """
#     Generate a Gold code sequence using two given polynomials and a specified length.
#     """
#     # Convert the polynomials from binary strings to lists of coefficients
#     poly1 = [int(c) for c in polynomial_1]
#     poly2 = [int(c) for c in polynomial_2]
#
#     # Initialize the shift registers to all ones
#     reg1 = [1] * len(poly1)
#     reg2 = [1] * len(poly2)
#
#     # Generate the Gold code sequence
#     gold = []
#     for i in range(length):
#         # Calculate the output bit as the XOR of the most significant bit of each register
#         output = reg1[0] ^ reg2[0]
#         gold.append(str(output))
#
#         # Shift the registers by one bit
#         reg1 = [output] + reg1[:-1]
#         reg2 = [reg1[-1] ^ sum([coeff * reg2[i] for i, coeff in enumerate(poly2[::-1])])] + reg2[:-1]
#
#     gold_code = "".join(gold)
#     return gold_code

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
        np.savetxt("Recording_handclap_" + str(i) + ".csv", mics[i], delimiter=",")
        return


def bit_string(length):
    bit_string = ""
    for i in range(length):
        bit = random.randint(0, 1)
        bit_string += str(bit)

    return bit_string


def gold_code(polynomial_1, polynomial_2, length):
    poly1 = [int(c) for c in polynomial_1]
    poly2 = [int(c) for c in polynomial_2]

    # convert polynomials to binary strings
    poly1_str = ''.join(str(bit) for bit in poly1)
    poly2_str = ''.join(str(bit) for bit in poly2)

    # set up LFSR registers
    reg1 = int(poly1_str, 2)
    reg2 = int(poly2_str, 2)

    gold = []
    for i in range(length):
        # XOR the outputs of the two registers
        output = (reg1 & 1) ^ (reg2 & 1)
        gold.append(output)

        # shift the registers to the right by 1 bit
        reg1 >>= 1
        reg2 >>= 1

        # apply feedback to the registers
        feedback1 = (reg1 >> (len(poly1) - 1)) ^ (reg1 & 1)
        feedback2 = (reg2 >> (len(poly2) - 1)) ^ (reg2 & 1)
        reg1 ^= feedback1 << (len(poly1) - 1)
        reg2 ^= feedback2 << (len(poly2) - 1)

    # convert the list of bits to a binary string
    gold_str = ''.join(str(bit) for bit in gold)

    return gold_str


poly1 = bit_string(200)
poly2 = bit_string(200)
length = 32
gold = gold_code(poly1, poly2, length)
# autocorr = autocorrelation_normalized([int(bit) for bit in gold])
print(gold)

gold_array = []
for i in range(len(gold)):
    gold_array.append(gold[i])
# print(gold_array)

gold_array_integer = [int(i) for i in gold_array]
# print(gold_array_integer)
r1 = np.convolve(gold_array_integer, np.flip(gold_array_integer))
n = list(range(-32 + 1, 32))
# print(r1.shape)
plt.plot(n, r1);
plt.xlabel('n')
plt.ylabel('r_xy[n]');
plt.show()

# bit_string = bit_string(20)
# hex_string = hex(int(bit_string, 2))


# gotten from https://github.com/mubeta06/python/blob/master/signal_processing/sp/gold.py
