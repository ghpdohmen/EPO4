import pyaudio as audio
import numpy as np
import matplotlib.pyplot as plt
import random
import robot

# import scipy
# pip install --upgrade --no-cache-dir gdown
# gdown 1xGUeeM-oY0pyXA0OO8_uKwT-vLAsiyZB  # refsignal.py
# gdown 1xTibH8tNbpwdSWmkziFGS24WXEYhtMRl  # wavaudioread.py
# gdown 10f3-zLIpu81jjQtr-Mr7BCtFz6u3o4N1 # recording_tool.py

# from scipy.io import wavfile
# from scipy.fft import fft, ifft
# from scipy.signal import convolve, unit_impulse
# from IPython.display import Audio

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

    from subsystems.subsystemStateEnum import subSystemState

    # class Localizationsubsystem(subSystem):
    #    enabled = False
    #
    # def __int__(self):
    #     # robot.speakerOn = False
    #     # robot.code = "EB3A994F"
    #     # robot.carrierFrequency = 6000
    #     # robot.bitFrequency = 2000
    #     # robot.repetitionCount = 64
    #     # robot.speakerOn = True
    #
    #
    #
    #
    # def start(self):
    #
    # def update(self):
    #
    # def stop(self):


def microphone_array(device_index, duration_recording):
    Fs = 44100
    number_of_samples = duration_recording * Fs

    pyaudio_handle = audio_devices(print_list=False)
    stream = pyaudio_handle.open(input_device_index=device_index, channels=5, format=audio.paInt16, rate=Fs, input=True)

    samples = stream.read(number_of_samples)
    data = np.frombuffer(samples, dtype='int16')

    data_length = len(data[::5])
    data_mic_0 = data[0::5]
    data_mic_1 = data[1::5]
    data_mic_2 = data[2::5]
    data_mic_3 = data[3::5]
    data_mic_4 = data[4::5]

    sample_axis_mic_0 = np.linspace(0, data_length, data_length)
    sample_axis_mic_1 = np.linspace(0, data_length, data_length)
    sample_axis_mic_2 = np.linspace(0, data_length, data_length)
    sample_axis_mic_3 = np.linspace(0, data_length, data_length)
    sample_axis_mic_4 = np.linspace(0, data_length, data_length)

    mic_0 = sample_axis_mic_0, data_mic_0
    mic_1 = sample_axis_mic_1, data_mic_1
    mic_2 = sample_axis_mic_2, data_mic_2
    mic_3 = sample_axis_mic_3, data_mic_3
    mic_4 = sample_axis_mic_4, data_mic_4

    return mic_0, mic_1, mic_2, mic_3, mic_4


def mic_plotter(data: bool, device_index=None, duration_recording=None):
    """
    @param data: True or False
    @param device_index: Integer
    @param duration_recording: Integer
    @return:
    """
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
    """
    @param device_index:
    @param duration_recording:
    @return:
    """
    mics = microphone_array(device_index, duration_recording)
    for i in range(5):
        np.savetxt("Recording_handclap_" + str(i) + ".csv", mics[i], delimiter=",")
        return


# def bit_string(length):
#     """
#     @param length:
#     @return:
#     """
#     bit_string = ""
#     for i in range(length):
#         bit = random.randint(0, 1)
#         bit_string += str(bit)
#
#     return bit_string


# def gold_code(polynomial_1, polynomial_2, length):
#     poly1 = [int(c) for c in polynomial_1]
#     poly2 = [int(c) for c in polynomial_2]
#
#     # convert polynomials to binary strings
#     poly1_str = ''.join(str(bit) for bit in poly1)
#     poly2_str = ''.join(str(bit) for bit in poly2)
#
#     # set up LFSR registers
#     reg1 = int(poly1_str, 2)
#     reg2 = int(poly2_str, 2)
#
#     gold = []
#     for i in range(length):
#         # XOR the outputs of the two registers
#         output = (reg1 & 1) ^ (reg2 & 1)
#         gold.append(output)
#
#         # shift the registers to the right by 1 bit
#         reg1 >>= 1
#         reg2 >>= 1
#
#         # apply feedback to the registers
#         feedback1 = (reg1 >> (len(poly1) - 1)) ^ (reg1 & 1)
#         feedback2 = (reg2 >> (len(poly2) - 1)) ^ (reg2 & 1)
#         reg1 ^= feedback1 << (len(poly1) - 1)
#         reg2 ^= feedback2 << (len(poly2) - 1)
#
#     # convert the list of bits to a binary string
#     gold_str = ''.join(str(bit) for bit in gold)
#
#     return gold_str


# def gold_code_generator(iterations, length_polynomials):
#     gold_code_array = []
#     cross_correlation = []
#     for i in range(iterations):
#         poly1 = bit_string(length_polynomials)
#         poly2 = bit_string(length_polynomials)
#         gold = gold_code(poly1, poly2, 32)
#         gold_code_array.append(gold)
#
#         gold_array = []
#         for j in range(len(gold)):
#             gold_array.append(gold[j])
#         # print(gold_array)
#
#         gold_array_integer = [int(k) for k in gold_array]
#         # print(gold_array_integer)
#
#         r = np.convolve(gold_array_integer, np.flip(gold_array_integer))
#         second_peak = max(r[32::])
#         difference = r[31] - second_peak
#         cross_correlation.append(difference)
#
#     maximum_difference = max(cross_correlation)
#     for i in range(len(gold_code_array)):
#         if maximum_difference == cross_correlation[i]:
#             gold_code_used = gold_code_array[i]
#             index = i
#
#     return gold_code_used, r, maximum_difference


# gold_code_used, r, maximum_difference = gold_code_generator(10000000, 200)
# print(gold_code_used, maximum_difference)
# n = list(range(-32 + 1, 32))
# plt.plot(n, r);
# plt.xlabel('n')
# plt.ylabel('r_xy[n]');
# plt.savefig("autocorrelation.png")
# plt.savefig("autocorrelation.pdf")
# plt.show()


# mic_plotter(False, 1, 0.032)
# # mic_plotter(False, 1, 0.064)
#
# for i in range(20):
#     device_index = 1
#     duration_recording = 0.032
#     robot.speakerOn = False
#     robot.code = "EB3A994F"
#     robot.carrierFrequency = 6000
#     robot.bitFrequency = 2000
#     robot.repetitionCount = 64
#     robot.speakerOn = True
#     mics = microphone_array(device_index, duration_recording)
#     for j in range(5):
#         np.savetxt("Recording_reference_" + str(j) + "_" + str(i) + ".csv", mics[i], delimiter=",")


# reference = np.empty([2, 176400])
# for i in range(20):
#     data = np.loadtxt("Recording_reference_0_" + str(i) + ".csv", delimiter=",")
#     reference[0] = data[0]
#     reference[1] = data[1] * 2
#
# reference[1] = reference[1]/20
# print(reference)
# plt.plot(reference[0], reference[1])
# plt.show()

# for j in range(10):
#     mics = microphone_array(1, 10)
#     for i in range(5):
#         np.savetxt("Recording_x,y_" + str(j) + str(i) + ".csv", mics[i], delimiter=",")

