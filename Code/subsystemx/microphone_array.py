import pyaudio as audio
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

import scipy.signal as sp
from scipy.linalg import pinv

from scipy.fft import fft, ifft
# from scipy.signal import butter, buttord, lfilter
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


def microphone_array(device_index, duration_recording):
    number_of_samples = duration_recording * Fs

    pyaudio_handle = audio_devices(print_list=False)
    stream = pyaudio_handle.open(input_device_index=device_index, channels=5, format=audio.paInt16, rate=Fs, input=True)

    samples = stream.read(number_of_samples)
    data = np.frombuffer(samples, dtype='int16')

    data_length = len(data[::5])
    data_mic_1 = data[0::5]
    data_mic_2 = data[1::5]
    data_mic_3 = data[2::5]
    data_mic_4 = data[3::5]
    data_mic_5 = data[4::5]

    sample_axis_mic_1 = np.linspace(0, data_length, data_length)
    sample_axis_mic_2 = np.linspace(0, data_length, data_length)
    sample_axis_mic_3 = np.linspace(0, data_length, data_length)
    sample_axis_mic_4 = np.linspace(0, data_length, data_length)
    sample_axis_mic_5 = np.linspace(0, data_length, data_length)

    mic_1 = sample_axis_mic_1, data_mic_1
    mic_2 = sample_axis_mic_2, data_mic_2
    mic_3 = sample_axis_mic_3, data_mic_3
    mic_4 = sample_axis_mic_4, data_mic_4
    mic_5 = sample_axis_mic_5, data_mic_5

    return mic_1, mic_2, mic_3, mic_4, mic_5


# def mic_plotter(data: bool, device_index=None, duration_recording=None):
#     """
#     @param data: True or False
#     @param device_index: Integer
#     @param duration_recording: Integer
#     @return:
#     """
#     if data is False:
#         mics = microphone_array(device_index, duration_recording)
#         for i in range(5):
#             plt.plot(mics[i][0], mics[i][1])
#             plt.show()
#         return
#     else:
#         for i in range(5):
#             data = np.loadtxt(
#                 r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Recording_reference_" + str(i) + "_1.csv",
#                 delimiter=",")
#             plt.plot(data[0], data[1])
#             plt.show()
#         return
#
#
# def reference_plotter():
#     for i in range(1, 6):
#         data = np.loadtxt(r"E:\TU Delft\Github\EPO4\Code\References\mic" + str(i) + "_reference_final.csv",
#                           delimiter=",")
#         plt.plot(data[0], data[1])
#         plt.title("Microphone " + str(i) + " reference")
#         plt.show()
#     return


# mic_plotter(True)


# def data_saver(device_index, duration_recording):
#     """
#     Generates a random bit string with a given length
#     @param _length: length of the bit string
#     @return: random bit string
#     """
#     _bit_string = ""
#     for i in range(_length):
#         _bit = random.randint(0, 1)
#         _bit_string += str(_bit)
#
#     return _bit_string
#
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

# def filtering(signal):
#     Fpass_lower = 4000
#     Fpass_higher = 8000
#     Fstop_lower = 3000
#     Fstop_higher = 9000
#     pass_damp = 3
#     stop_damp = 40
#
#     N, Wn = buttord([Fpass_lower / Fs * 2, Fpass_higher / Fs * 2], [Fstop_lower / Fs * 2, Fstop_higher / Fs * 2],
#                     pass_damp, stop_damp)
#     b, a = butter(N, Wn, btype='bandpass')
#
#     filtered_signal = np.zeros((2, len(signal[0])))
#     filtered_signal[0] = signal[0]
#     filtered_signal[1] = lfilter(b, a, signal[1])
#     return filtered_signal


def ch3(y):
    epsi = 0.02
    signal_reference = reference_array()
    Nx = len(signal_reference[1])  # Length of x
    Ny = len(y)  # Length of y
    L = Ny - Nx + 1  # Length of h

    x = np.concatenate((signal_reference[1], np.zeros(L - 1)))

    # Deconvolution in frequency domain
    Y = fft(y)
    X = fft(x, Ny)
    H = Y / (X + 10e-15)

    # Threshold to avoid blow ups of noise during inversion
    ii = np.absolute(X) < epsi * max(abs(X))
    H[ii] = 0

    h = np.real(ifft(H))  # ensure the result is real

    return h


def reference_array():
    reference_mic = np.loadtxt(
        r"E:\TU Delft\Github\EPO4\Code\References\mic1_reference_final.csv",
        delimiter=',')
    return reference_mic


def isolation(recorded_signal):
    reference_signal = reference_array()
    correlation = sp.correlate(recorded_signal[1], reference_signal[1], mode='same')
    # plt.plot(correlation)
    # plt.title("Correlation of Reference and Recorded Signals")
    # plt.xlabel("Samples")
    # plt.ylabel("Amplitude")
    # # plt.savefig("Correlation.png")
    # plt.show()

    peak_index, = sp.argrelmax(correlation, order=1000)
    # print(peak_index)
    if (peak_index[0] < 500):
        peak_index = np.delete(peak_index, [0], None)
    # print(peak_index[0])
    if len(peak_index) == 2:
        pulse_delay = peak_index[0] - (len(reference_signal[1]) // 2)
    else:
        pulse_delay = peak_index[1] - (len(reference_signal[1]) // 2)

    # pulse_delay = peak_index[0] - (len(reference_signal[1]) // 2)


    # threshold = 0.9 * max(correlation)
    # peak_index = sp.find_peaks(correlation, threshold, distance=1400)
    # pulse_delay = peak_index[0] - (len(reference_signal[1]) // 2)

    # peak_index, = np.where(correlation == max(correlation))
    # pulse_delay = int(peak_index - (len(reference_signal[1]) // 2))

    isolated_pulse = np.zeros((2, len(reference_signal[0])))
    isolated_pulse[0] = recorded_signal[0][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]
    isolated_pulse[1] = recorded_signal[1][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]
    # plt.plot(isolated_pulse[0], isolated_pulse[1])
    # plt.title("Isolated Second Pulse")
    # plt.xlabel("Samples")
    # plt.ylabel("Amplitude")
    # # plt.savefig("isolated_pulse.png")
    # plt.show()
    return isolated_pulse


def tdoa(signal_recorded_1, signal_recorded_2, signal_recorded_3, signal_recorded_4, signal_recorded_5):
    isolated_pulse_mic_1 = isolation(signal_recorded_1)
    zeros_1 = np.zeros(int(isolated_pulse_mic_1[0, 0]))
    channel_signal_1 = np.zeros((2, math.ceil(isolated_pulse_mic_1[0, -1])))
    channel_signal_1[0] = np.concatenate((zeros_1, isolated_pulse_mic_1[0]))
    channel_signal_1[1] = np.concatenate((zeros_1, isolated_pulse_mic_1[1]))
    channel_1 = ch3(channel_signal_1[1])
    # plt.plot(channel_1)
    # plt.show()
    maximum_1, = np.where(abs(channel_1) == max(abs(channel_1)))

    isolated_pulse_mic_2 = isolation(signal_recorded_2)
    zeros_2 = np.zeros(int(isolated_pulse_mic_2[0, 0]))
    channel_signal_2 = np.zeros((2, math.ceil(isolated_pulse_mic_2[0, -1])))
    channel_signal_2[0] = np.concatenate((zeros_2, isolated_pulse_mic_2[0]))
    channel_signal_2[1] = np.concatenate((zeros_2, isolated_pulse_mic_2[1]))
    channel_2 = ch3(channel_signal_2[1])
    # plt.plot(channel_2)
    # plt.show()
    maximum_2, = np.where(abs(channel_2) == max(abs(channel_2)))

    isolated_pulse_mic_3 = isolation(signal_recorded_3)
    zeros_3 = np.zeros(int(isolated_pulse_mic_3[0, 0]))
    channel_signal_3 = np.zeros((2, math.ceil(isolated_pulse_mic_3[0, -1])))
    channel_signal_3[0] = np.concatenate((zeros_3, isolated_pulse_mic_3[0]))
    channel_signal_3[1] = np.concatenate((zeros_3, isolated_pulse_mic_3[1]))
    channel_3 = ch3(channel_signal_3[1])
    # plt.plot(channel_3)
    # plt.show()
    maximum_3, = np.where(abs(channel_3) == max(abs(channel_3)))

    isolated_pulse_mic_4 = isolation(signal_recorded_4)
    zeros_4 = np.zeros(int(isolated_pulse_mic_4[0, 0]))
    channel_signal_4 = np.zeros((2, math.ceil(isolated_pulse_mic_4[0, -1])))
    channel_signal_4[0] = np.concatenate((zeros_4, isolated_pulse_mic_4[0]))
    channel_signal_4[1] = np.concatenate((zeros_4, isolated_pulse_mic_4[1]))
    channel_4 = ch3(channel_signal_4[1])
    # plt.plot(channel_4)
    # plt.show()
    maximum_4, = np.where(abs(channel_4) == max(abs(channel_4)))

    isolated_pulse_mic_5 = isolation(signal_recorded_5)
    zeros_5 = np.zeros(int(isolated_pulse_mic_5[0, 0]))
    channel_signal_5 = np.zeros((2, math.ceil(isolated_pulse_mic_5[0, -1])))
    channel_signal_5[0] = np.concatenate((zeros_5, isolated_pulse_mic_5[0]))
    channel_signal_5[1] = np.concatenate((zeros_5, isolated_pulse_mic_5[1]))
    channel_5 = ch3(channel_signal_5[1])
    # plt.plot(channel_5)
    # plt.show()
    maximum_5, = np.where(abs(channel_5) == max(abs(channel_5)))

    # r12, r13, r14, r15, r23, r24, r25, r34, r35, r45
    distance = np.zeros(10)
    distance[0] = maximum_1 - maximum_2
    distance[1] = maximum_1 - maximum_3
    distance[2] = maximum_1 - maximum_4
    distance[3] = maximum_1 - maximum_5
    distance[4] = maximum_2 - maximum_3
    distance[5] = maximum_2 - maximum_4
    distance[6] = maximum_2 - maximum_5
    distance[7] = maximum_3 - maximum_4
    distance[8] = maximum_3 - maximum_5
    distance[9] = maximum_4 - maximum_5

    maximum = np.zeros(5)
    maximum[0] = maximum_1
    maximum[1] = maximum_2
    maximum[2] = maximum_3
    maximum[3] = maximum_4
    maximum[4] = maximum_5

    time = np.zeros(10)
    distance_cm = np.zeros(10)
    for i in range(10):
        time[i] = distance[i] / Fs
        distance_cm[i] = time[i] * 34300
    # print(distance)
    # print(maximum)
    return distance
    # return maximum


signal_recorded_1 = np.loadtxt(
    r"E:\TU Delft\Github\EPO4\Code\Square\Recording_73x80.5_test_1_1.csv",
    delimiter=',')

# plt.plot(signal_recorded_1[0], signal_recorded_1[1])
# plt.show()

signal_recorded_2 = np.loadtxt(
    r"E:\TU Delft\Github\EPO4\Code\Square\Recording_73x80.5_test_1_2.csv",
    delimiter=',')

# plt.plot(signal_recorded_2[0], signal_recorded_2[1])
# plt.show()

signal_recorded_3 = np.loadtxt(
    r"E:\TU Delft\Github\EPO4\Code\Square\Recording_73x80.5_test_1_3.csv",
    delimiter=',')

# plt.plot(signal_recorded_3[0], signal_recorded_3[1])
# plt.show()

signal_recorded_4 = np.loadtxt(
    r"E:\TU Delft\Github\EPO4\Code\Square\Recording_73x80.5_test_1_4.csv",
    delimiter=',')

# plt.plot(signal_recorded_4[0], signal_recorded_4[1])
# plt.show()

signal_recorded_5 = np.loadtxt(
    r"E:\TU Delft\Github\EPO4\Code\Square\Recording_73x80.5_test_1_5.csv",
    delimiter=',')

# plt.plot(signal_recorded_5[0], signal_recorded_5[1])
# plt.show()


reference_signal = reference_array()
sp.correlate(signal_recorded_1[1], reference_signal[1], mode='same')

sp.correlate(signal_recorded_2[1], reference_signal[1], mode='same')

sp.correlate(signal_recorded_3[1], reference_signal[1], mode='same')


sp.correlate(signal_recorded_4[1], reference_signal[1], mode='same')


sp.correlate(signal_recorded_5[1], reference_signal[1], mode='same')

# distance = [-306.21, -461.64, -297.44, -66.74, -155.43, 8.77, 239.47, 164.2, 394.9, 230.7]
def estimate_location(distance):
    coordinates_mics = np.array([[0, 480], [480, 480], [480, 0], [0, 0], [0, 240]])
    # Create indexes for all microphone pairs
    pairs = list(
        itertools.combinations([1, 2, 3, 4, 5], 2))  # r12, r13, r14, r15, r23, r24, r25, r34, r35, r45

    A = np.zeros((10, 6))
    B = np.zeros((10, 1))

    first_column = np.zeros((10, 2))
    for row, [i, j] in enumerate(pairs):
        first_column[row] = (2 * (coordinates_mics[j-1] - coordinates_mics[i-1]))
        A[row, 0:2] = first_column[row]

        A[row, j] = -2 * distance[row]

        B[row] = pow(distance[row], 2) - pow(np.linalg.norm(coordinates_mics[i-1]), 2) + pow(np.linalg.norm(coordinates_mics[j-1]), 2)

    A_pinv = np.linalg.pinv(A)

    y = np.dot(A_pinv, B)

    xy = np.squeeze(y[0:2])
    print(xy)
    return(xy)

# estimate_location(distance)
estimate_location(tdoa(signal_recorded_1, signal_recorded_2, signal_recorded_3, signal_recorded_4, signal_recorded_5))
