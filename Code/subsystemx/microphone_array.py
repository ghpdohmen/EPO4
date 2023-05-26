import pyaudio as audio
import numpy as np
import matplotlib.pyplot as plt
import robot
import math
import itertools

import scipy.signal as sp
from scipy.linalg import pinv

# pip install --upgrade --no-cache-dir gdown
# gdown 1xGUeeM-oY0pyXA0OO8_uKwT-vLAsiyZB  # refsignal.py
# gdown 1xTibH8tNbpwdSWmkziFGS24WXEYhtMRl  # wavaudioread.py
# gdown 10f3-zLIpu81jjQtr-Mr7BCtFz6u3o4N1 # recording_tool.py

# from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import butter, buttord, lfilter
from scipy.signal import convolve, unit_impulse
# from IPython.display import Audio

from subsystemx.subsystem import subSystem

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
            data = np.loadtxt(
                r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Recording_reference_" + str(i) + "_1.csv",
                delimiter=",")
            plt.plot(data[0], data[1])
            plt.show()
        return


def reference_plotter():
    for i in range(1, 6):
        data = np.loadtxt(r"E:\TU Delft\Github\EPO4\Code\References\mic" + str(i) + "_reference_final.csv",
                          delimiter=",")
        plt.plot(data[0], data[1])
        plt.title("Microphone " + str(i) + " reference")
        plt.show()
    return


# mic_plotter(True)


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


# def bit_string(_length):
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
# #
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
#
#
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

# 11101011001110101001100101001111
# gold_code_used, r, maximum_difference = gold_code_generator(10000000, 200)
# print(gold_code_used, maximum_difference)
# n = list(range(-32 + 1, 32))
# plt.plot(n, r);
# plt.xlabel('n')
# plt.ylabel('r_xy[n]');
# plt.savefig("autocorrelation.png")
# plt.savefig("autocorrelation.pdf")
# plt.show()


# for i in range(5):
#     data = np.loadtxt(r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Recording_reference_1_" + str(i) + ".csv", delimiter=",")
# for i in range(1, 5):
#     data = np.loadtxt(r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\Recording_reference_mic1_" + str(i) + ".csv", delimiter=",")
#     plt.plot(data[0], data[1])
#     # plt.title("Microphone " + str(i+1))
#     plt.title("microphone 1, recording " + str(i))
#     # plt.xlim(50, 500)
#     plt.show()

#

#
# plt.plot(data[0], data[1])
# plt.title("microphone 5, recording 3")
# plt.xlim(3019, 3730)
# plt.show()

# def truncater(start, stop):
#     data = np.loadtxt(
#         r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\Recording_reference_mic5_3.csv",
#         delimiter=",")
#     data_truncated = data[0][start:stop], data[1][start:stop]
#     np.savetxt(r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic5_3.csv", data_truncated,
#                delimiter=",")
#     return
#

#
# truncater(3019, 3730)
# data = np.loadtxt(r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic5_3.csv",
#                   delimiter=",")
# plt.plot(data[0], data[1])
# plt.title("microphone5, reference 3")
# plt.xlim(3019, 3730)
# plt.show()
# print(data.shape)

# reference = np.zeros((2, 711))
# for i in range(1, 3):
#     data = np.loadtxt(r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic5_" + str(i) + ".csv", delimiter=",")
#     reference[0] = np.linspace(0, 711, 711)
#     reference[1] = np.add(reference[1], data[1])
#     # print(reference[1])
#
# reference[1] = reference[1]/3
# reference[1] = reference[1]/np.max(reference[1])
# # print(np.max(reference[1]))
# plt.plot(reference[0], reference[1])
# plt.xlim(0, 711)
# plt.show()
# # # #
# np.savetxt(r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic5_reference_final.csv", reference,
#                delimiter=",")


# for i in range(5):
#     data = np.loadtxt(r"E:\TU Delft\Github\EPO4\Code\Square\Recording_middle_1_" + str(i) + ".csv", delimiter=",")
#     plt.plot(data[0], data[1])
#     plt.title("480x480, Microphone " + str(i) + " reference")
#     plt.show()


def filtering(signal):
    Fpass_lower = 4000
    Fpass_higher = 8000
    Fstop_lower = 3000
    Fstop_higher = 9000
    pass_damp = 3
    stop_damp = 40

    N, Wn = buttord([Fpass_lower / Fs * 2, Fpass_higher / Fs * 2], [Fstop_lower / Fs * 2, Fstop_higher / Fs * 2],
                    pass_damp, stop_damp)
    b, a = butter(N, Wn, btype='bandpass')

    filtered_signal = np.zeros((2, len(signal[0])))
    filtered_signal[0] = signal[0]
    filtered_signal[1] = lfilter(b, a, signal[1])
    return filtered_signal


def ch3(y):
    epsi = 0.02
    signal_reference = reference_array()
    Nx = len(signal_reference[1])  # Length of x
    Ny = len(y)  # Length of y
    L = Ny - Nx + 1  # Length of h

    # len(x) == len(y)
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
        r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic1_reference_final.csv",
        delimiter=',')
    return reference_mic


def isolation(recorded_signal):
    reference_signal = reference_array()
    correlation = sp.correlate(recorded_signal[1], reference_signal[1], mode='same')
    # plt.plot(correlation)
    # plt.title("correlation")
    # plt.show()

    peak_index, = sp.argrelmax(correlation, order=1000)
    # print(peak_index)
    if (peak_index[0] < 500 or peak_index[0] > 6300):
        peak_index = np.delete(peak_index, [0], None)
    # print(peak_index)
    if len(peak_index) == 2:
        pulse_delay = peak_index[0] - (len(reference_signal[1]) // 2)
    else:
        pulse_delay = peak_index[1] - (len(reference_signal[1]) // 2)

    isolated_pulse = np.zeros((2, len(reference_signal[0])))
    isolated_pulse[0] = recorded_signal[0][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]
    isolated_pulse[1] = recorded_signal[1][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]
    return isolated_pulse


def tdoa(signal_recorded_1, signal_recorded_2, signal_recorded_3, signal_recorded_4, signal_recorded_5):
    isolated_pulse_mic_1 = isolation(signal_recorded_1)
    zeros_1 = np.zeros(int(isolated_pulse_mic_1[0][0]))
    channel_signal_1 = np.zeros((2, math.ceil(isolated_pulse_mic_1[0][-1])))
    channel_signal_1[0] = np.concatenate((zeros_1, isolated_pulse_mic_1[0]))
    channel_signal_1[1] = np.concatenate((zeros_1, isolated_pulse_mic_1[1]))
    channel_1 = ch3(channel_signal_1[1])
    maximum_1, = np.where(abs(channel_1) == max(abs(channel_1)))

    isolated_pulse_mic_2 = isolation(signal_recorded_2)
    zeros_2 = np.zeros(int(isolated_pulse_mic_2[0][0]))
    channel_signal_2 = np.zeros((2, math.ceil(isolated_pulse_mic_2[0][-1])))
    channel_signal_2[0] = np.concatenate((zeros_2, isolated_pulse_mic_2[0]))
    channel_signal_2[1] = np.concatenate((zeros_2, isolated_pulse_mic_2[1]))
    channel_2 = ch3(channel_signal_2[1])
    maximum_2, = np.where(abs(channel_2) == max(abs(channel_2)))

    isolated_pulse_mic_3 = isolation(signal_recorded_3)
    zeros_3 = np.zeros(int(isolated_pulse_mic_3[0][0]))
    channel_signal_3 = np.zeros((2, math.ceil(isolated_pulse_mic_3[0][-1])))
    channel_signal_3[0] = np.concatenate((zeros_3, isolated_pulse_mic_3[0]))
    channel_signal_3[1] = np.concatenate((zeros_3, isolated_pulse_mic_3[1]))
    channel_3 = ch3(channel_signal_3[1])
    maximum_3, = np.where(abs(channel_3) == max(abs(channel_3)))

    isolated_pulse_mic_4 = isolation(signal_recorded_4)
    zeros_4 = np.zeros(int(isolated_pulse_mic_4[0][0]))
    channel_signal_4 = np.zeros((2, math.ceil(isolated_pulse_mic_4[0][-1])))
    channel_signal_4[0] = np.concatenate((zeros_4, isolated_pulse_mic_4[0]))
    channel_signal_4[1] = np.concatenate((zeros_4, isolated_pulse_mic_4[1]))
    channel_4 = ch3(channel_signal_4[1])
    maximum_4, = np.where(abs(channel_4) == max(abs(channel_4)))

    isolated_pulse_mic_5 = isolation(signal_recorded_5)
    zeros_5 = np.zeros(int(isolated_pulse_mic_5[0][0]))
    channel_signal_5 = np.zeros((2, math.ceil(isolated_pulse_mic_5[0][-1])))
    channel_signal_5[0] = np.concatenate((zeros_5, isolated_pulse_mic_5[0]))
    channel_signal_5[1] = np.concatenate((zeros_5, isolated_pulse_mic_5[1]))
    channel_5 = ch3(channel_signal_5[1])
    maximum_5, = np.where(abs(channel_5) == max(abs(channel_5)))

    # r12, r13, r14, r15, r23, r24, r25, r34, r35, r45
    distance = np.zeros(10)
    distance[0] = abs(maximum_1 - maximum_2)
    distance[1] = abs(maximum_1 - maximum_3)
    distance[2] = abs(maximum_1 - maximum_4)
    distance[3] = abs(maximum_1 - maximum_5)
    distance[4] = abs(maximum_2 - maximum_3)
    distance[5] = abs(maximum_2 - maximum_4)
    distance[6] = abs(maximum_2 - maximum_5)
    distance[7] = abs(maximum_3 - maximum_4)
    distance[8] = abs(maximum_3 - maximum_5)
    distance[9] = abs(maximum_4 - maximum_5)

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
    print(distance_cm)
    # return distance_cm
    return maximum


def estimate_location(rij):
    microphone_locations = np.array([[0, 480], [480, 480], [480, 0], [0, 0], [0, 240]])
    num_mics = microphone_locations.shape[0]  # number of microphones

    # Construct the matrix A
    A = np.zeros((num_mics * (num_mics - 1) // 2, num_mics + 1))  # create zero matrix with the correct shape
    row = 0

    # loop over microphone pairs
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            x_diff = 2 * (microphone_locations[j] - microphone_locations[i]).T

            A[row, 0] = x_diff[0]  # x-value
            A[row, 1] = x_diff[1]  # y-value
            A[row, j] = -2 * rij[row]  # range difference between microphones

            # Assign zero to every column except 0 and j
            for k in range(num_mics):
                if k != 0 and k != 1 and k != j:
                    A[row, k] = 0

            row += 1

    # Construct the matrix b
    b = np.zeros((num_mics * (num_mics - 1) // 2, 1))  # has form of 10 x 1
    row = 0

    # loop over microphone pairs
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            xi_norm_squared = np.linalg.norm(microphone_locations[i]) ** 2  # Extract and normalize
            xj_norm_squared = np.linalg.norm(microphone_locations[j]) ** 2

            b[row] = rij[row] ** 2 - xi_norm_squared + xj_norm_squared

            row += 1

    # Solve for x and d
    A_inv = pinv(A)  # pseudo-inverse
    x_d = A_inv @ b
    x = x_d[:2]  # select the first two elements from the array (x, y)
    d = x_d[2:]  # select from the third element to the end
    print(x)
    return x, d


#
signal_recorded_1 = np.loadtxt(
    r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_137x162_test_3_1.csv",
    delimiter=',')

signal_recorded_2 = np.loadtxt(
    r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_137x162_test_3_2.csv",
    delimiter=',')

signal_recorded_3 = np.loadtxt(
    r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_137x162_test_3_3.csv",
    delimiter=',')
signal_recorded_4 = np.loadtxt(
    r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_137x162_test_3_4.csv",
    delimiter=',')
signal_recorded_5 = np.loadtxt(
    r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_137x162_test_3_5.csv",
    delimiter=',')


# tdoa(signal_recorded_1, signal_recorded_2, signal_recorded_3, signal_recorded_4, signal_recorded_5)


def localize(maximum):
    coordinates_mics = np.array([[0, 480], [480, 480], [480, 0], [0, 0], [0, 240]])
    # Create indexes for all microphone pairs
    pairs = list(
        itertools.combinations([0, 1, 2, 3, 4], 2))  # r12, r13, r14, r15, r23, r24, r25, r34, r35, r45

    # Create the empty matrices, +1 for the vector
    A = np.zeros([len(pairs), len(coordinates_mics) + 1])
    B = np.zeros([len(pairs), 1])

    # Fill matrix
    for row, [i, j] in enumerate(pairs):

        # Calculate differences between peaks
        r_ij = maximum[i] - maximum[j]

        # Matrix A, column 1
        A[row, :2] = np.array(2 * (coordinates_mics[j] - coordinates_mics[i]))

        # Rest of matrix A
        A[row, j + 1] = -2 * r_ij

        # B matrix
        B[row] = r_ij ** 2 - np.linalg.norm(coordinates_mics[i]) ** 2 + np.linalg.norm(
            coordinates_mics[j]) ** 2,

    # Ay = B
    # Pseudo-inverse of A
    A_pinv = np.linalg.pinv(A)

    y = np.dot(A_pinv, B)
    x = y[:2].squeeze()  # Remove a dimension and use only the first two values (which is the vector)

    xy = x  # in [x,y]
    print(xy)
    return xy


localize(tdoa(signal_recorded_1, signal_recorded_2, signal_recorded_3, signal_recorded_4, signal_recorded_5))
