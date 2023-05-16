import pyaudio as audio
import numpy as np
import matplotlib.pyplot as plt
import robot
import math

import scipy.signal as sp
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


# TODO: write this as a subsystem

# from subsystemx.subsystemStateEnum import subSystemState

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
# data = np.loadtxt(r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Recording_reference_0_0.csv", delimiter=",")


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


# def bit_string(self, _length):
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

# def gold_code(self, _polynomial_1, _polynomial_2, _length):
#     """
#
#     @param _polynomial_1: First polynomial with random bit string
#     @param _polynomial_2: Second polynomial with random bit string
#     @param _length: length of the code
#     @return: into a string
#     """
#     _poly1 = [int(c) for c in _polynomial_1]
#     _poly2 = [int(c) for c in _polynomial_2]
#
#     # convert polynomials to binary strings
#     _poly1_str = ''.join(str(bit) for bit in _poly1)
#     _poly2_str = ''.join(str(bit) for bit in _poly2)
#
#     # set up LFSR registers
#     _reg1 = int(_poly1_str, 2)
#     _reg2 = int(_poly2_str, 2)
#
#     _gold = []
#     for i in range(_length):
#         # XOR the outputs of the two registers
#         _output = (_reg1 & 1) ^ (_reg2 & 1)
#         _gold.append(_output)
#
#         # shift the registers to the right by 1 bit
#         _reg1 >>= 1
#         _reg2 >>= 1
#
#         # apply feedback to the registers
#         _feedback1 = (_reg1 >> (len(_poly1) - 1)) ^ (_reg1 & 1)
#         _feedback2 = (_reg2 >> (len(_poly2) - 1)) ^ (_reg2 & 1)
#         _reg1 ^= _feedback1 << (len(_poly1) - 1)
#         _reg2 ^= _feedback2 << (len(_poly2) - 1)
#
#     # convert the list of bits to a binary string
#     _gold_str = ''.join(str(_bit) for _bit in _gold)
#     print(_gold_str)
#
#     return _gold_str
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

def truncater(start, stop):
    data = np.loadtxt(
        r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\Recording_reference_mic5_3.csv",
        delimiter=",")
    data_truncated = data[0][start:stop], data[1][start:stop]
    np.savetxt(r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic5_3.csv", data_truncated,
               delimiter=",")
    return


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


def ch3(x, y, epsi):
    Nx = len(x)  # Length of x
    Ny = len(y)  # Length of y
    L = Ny - Nx + 1  # Length of h

    # len(x) == len(y)
    x = np.concatenate((x, np.zeros(L - 1)))

    # Deconvolution in frequency domain
    Y = fft(y)
    X = fft(x, Ny)
    H = Y / (X + 10e-15)
    # Threshold to avoid blow ups of noise during inversion
    ii = np.absolute(X) < epsi * max(abs(X))
    H[ii] = 0

    h = np.real(ifft(H))  # ensure the result is real
    # h = h[0:34300]        # optional: truncate to length Lhat (L is not reliable?)

    return h


# signal_reference = np.loadtxt(
#         r"E:\TU Delft\Github\EPO4\Code\References\mic5_reference_final.csv",
#         delimiter=",")
# signal_reference = signal_reference[1]
#
# signal_recorded = np.loadtxt(
#         r"E:\TU Delft\Github\EPO4\Code\Square\Recording_middle_1_4.csv",
#         delimiter=",")

# plt.plot(signal_recorded[0], signal_recorded[1])
# plt.xlim(0, 2000)
# plt.show()
def tdoa(signal_reference, signal_recorded, min_value):
    signal_reference = signal_reference[1]
    # find index of, and maximum value of reference signal
    period = 1 / Fs
    begin = 0
    end = 2000
    slices = slice(begin, end, 1)
    max_value = max(abs(signal_recorded[1][slices]));

    for i in range(0, len(signal_recorded[0]), 2000):
        if end < len(signal_recorded[0]):
            begin += 2000
            end += 2000
            slices = slice(begin, end, 1)
            if max(abs(signal_recorded[1][slices])) > max_value:
                max_value = max(abs(signal_recorded[1][slices]));
                max_value_reference, = np.where(abs(signal_recorded[1][slices]) == max_value);
                max_value_reference = max_value_reference + begin;
                max_value_reference = np.append(max_value_reference, max_value);
                if max_value > 0.8 * max(abs(signal_recorded[1])):
                    break
        else:
            break

    # cut reference and recorded signal to one pulse + same length
    begin = int(max_value_reference[0] - 800)
    end = int(max_value_reference[0] + 1000)

    signal_recorded_used = np.zeros((2, end - begin))
    signal_recorded_used[0] = signal_recorded[0][begin:end]
    signal_recorded_used[1] = signal_recorded[1][begin:end]

    # estimate channel
    channel = ch3(signal_reference, signal_recorded_used[1], 0.01)
    channel_half = np.array_split(channel, 2)
    # middle = int(len(channel)/2)
    # print(channel_half[0].shape)
    # channel_half = channel[:middle]
    # find index of maximum value of channel
    max_value_channel, = np.where(abs(channel_half[0]) == max(abs(channel_half[0])))
    # calculate the corresponding amount of time
    time = period * max_value_channel
    # print(time)

    # calculate distance between microphones
    distance = time * 34300

    return channel, signal_reference, signal_recorded_used, distance, channel_half[0]
    # return distance


#
# signal_reference = np.loadtxt(r"E:\TU Delft\Github\EPO4\Code\References\mic5_reference_final.csv", delimiter=',')
# signal_recorded = np.loadtxt(r"E:\TU Delft\Github\EPO4\Code\Square\Recording_middle_1_4.csv", delimiter=',')

# signal_used = np.zeros((2,1420))
# signal_used[0] = signal_recorded[0][2050:3470]
# signal_used[1] = signal_recorded[1][2050:3470]

# distance = tdoa(signal_reference, signal_recorded, 0.02)
# print(distance)
# channel, signal_reference_1, signal_recorded_1, distance, channel_half = tdoa(signal_reference, signal_recorded, 0.01)
# channel, signal_reference_1, signal_recorded_1, distance, channel_half = tdoa(signal_used, signal_recorded, 0.01)

# print(signal_reference_1.shape, signal_recorded_1.shape, signal_reference.shape, signal_recorded.shape)
# channel = ch3(signal_reference[1], signal_recorded[1], 0.01)
# channel_taken = channel[:1000]
# max_channel, = np.where(abs(channel) == max(abs(channel_taken)))
# distance = max_channel/Fs*34300
# print(max_channel)


# channel = ch3(signal_reference[1], signal_recorded[1], 0.01)
# indexes, = np.where(abs(channel[0:5000]) > 60)
# print(indexes)
# # difference = indexes[3] - indexes[2]
# # print(difference)
# time = (indexes[5]-1650) * 1/Fs
# distance = time * 34300
# print(distance)
# print(channel.shape)

# print(distance)
# plt.plot(signal_reference[0], signal_reference[1])
# plt.title("Reference recording microphone 2")
# plt.show()


signal_reference = np.loadtxt(
    r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic3_reference_final.csv", delimiter=',')
signal_recorded = np.loadtxt(r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_middle_2_2.csv",
                             delimiter=',')

signal_reference_padded = np.zeros((2, 2 * len(signal_reference[0])))
# test = np.linspace(0, len(signal_reference_padded[0]), len(signal_reference_padded[0]))
# print(test)
signal_reference_padded[0] = np.linspace(0, len(signal_reference_padded[0]), len(signal_reference_padded[0]))
signal_reference_padded[1] = np.concatenate((signal_reference[1], np.zeros(len(signal_reference[1]))))


# plt.plot(signal_reference_padded[0], signal_reference_padded[1])
# plt.show()

# Y = fft(signal_reference[1])
# freq = np.linspace(0, Fs, len(Y))


# plt.plot(freq, abs(Y))
# plt.xlim(0, 10000)
# plt.show()

def filtering(signal):
    Fpass_lower = 4000
    Fpass_higher = 7000
    Fstop_lower = 3000
    Fstop_higher = 8000
    pass_damp = 3
    stop_damp = 40

    # N, Wn = buttord(Fpass / Fs * 2, Fstop / Fs * 2, pass_damp, stop_damp)
    # b, a = butter(N, Fpass / Fs * 2)
    N, Wn = buttord([Fpass_lower / Fs * 2, Fpass_higher / Fs * 2], [Fstop_lower / Fs * 2, Fstop_lower / Fs * 2],
                    pass_damp, stop_damp)
    b, a = butter(N, Wn, btype='bandpass')

    # impulse response:
    # h = lfilter(b, a, np.concatenate(([1], np.zeros(99))))
    # H = fft(h)
    # freq_1 = np.linspace(0, Fs, len(H))
    # plt.plot(freq_1, abs(H))
    # plt.show()

    filtered_signal = np.zeros((2, len(signal[0])))
    filtered_signal[0] = signal[0]
    filtered_signal[1] = lfilter(b, a, signal[1])
    return filtered_signal


signal_recorded_filtered = filtering(signal_recorded)
# plt.plot(signal_recorded_filtered[0], signal_recorded_filtered[1])
# plt.xlim(0, 2000)
# plt.show()

maxima, = sp.argrelmax(signal_recorded[1], order=800)
# print(maxima, maxima[1] - maxima[0])

truncation = np.zeros((2, maxima[1] - maxima[0]))
# # # # truncation[0] = np.concatenate((np.zeros(maxima[0]), signal_recorded[0][maxima[0]:maxima[1]]))
# # # # truncation[1] = np.concatenate((np.zeros(maxima[0]), signal_recorded[1][maxima[0]:maxima[1]]))
truncation[0] = signal_recorded[0][(maxima[0] - 130):(maxima[1] - 130)]
truncation[1] = signal_recorded[1][(maxima[0] - 130):(maxima[1] - 130)]

# signal_reference_padded[0] = np.linspace(0, len(signal_reference_padded[0]), len(signal_reference_padded[0]))
# signal_reference_padded[1] = np.concatenate((signal_reference[1], np.zeros(len(signal_reference[1]))))

truncation_padded = np.zeros((2, math.ceil(max(truncation[0]))))
truncation_padded[0] = np.linspace(0, len(truncation_padded[0]), len(truncation_padded[0]))
truncation_padded[1] = np.concatenate((np.zeros(int(truncation[0][0])), truncation[1]))

plt.plot(truncation_padded[0], truncation_padded[1])
plt.show()

channel = ch3(signal_reference_padded[1], truncation_padded[1], 0.01)
plt.plot(channel)
plt.show()

# plt.plot(truncation[0], truncation[1])
# plt.title("Truncated Signal")
# plt.show()

# # print(truncation.shape)
# truncation_1 = np.zeros((2, maxima[1]-300))
# truncation_1[0] = np.concatenate((np.zeros(int(truncation[0][0])), truncation[0]))
# truncation_1[1] = np.concatenate((np.zeros(int(truncation[0][0])), truncation[1]))
#
# plt.plot(truncation_1[0], truncation_1[1])
# plt.title("Truncated_1 Signal")
# plt.show()
#
# # # minima, = sp.argrelmin(truncation[1], order=800)
# # # print(minima+maxima[0])
# #
# # window_start = max(0, maxima[1] - 1400 // 2)
# # window_end = min(len(signal_recorded[1]), maxima[1] + 1400 // 2)
# #
# # isolated_pulse = signal_recorded[1][window_start:window_end]
# # t = np.linspace(0, len(isolated_pulse), len(isolated_pulse))
# # print(isolated_pulse)
#
# # plt.plot(t, isolated_pulse)
# # plt.show()
#
# channel = ch3(signal_reference[1], signal_recorded[1], 0.01)
# channel = ch3(signal_reference[1], truncation_1[1], 0.01)
# plt.plot(channel)
# plt.show()
# #
maximum, = np.where(abs(channel) == max(abs(channel)))
# print(maximum)
time = maximum * 1 / Fs
distance = time * 34300
print(distance)


# mic1: -800, +1000
# mic2: -550, +600 or -1100, +100

# signal_reference = np.loadtxt(r"E:\TU Delft\Github\EPO4\Code\References\Recording_reference_mic1_1.csv", delimiter=',')
# plt.plot(signal_reference[0], signal_reference[1])
# plt.show()


# multilateration TDOA (matrix)
import numpy as np
from scipy.linalg import pinv


def solve_xandD(A, b):
    A_inv = pinv(A)  # pseudo-inverse
    x_d = A_inv @ b

    # Microphone locations
    microphone_locations = np.array([[0, 480], [480, 480], [480, 0], [0, 0], [0, 240]])

    num_mics = microphone_locations.shape[0]  # number of microphones

    # Measured range differences (TDOA * speed of sound)
    # rij = np.array([r12, r13, r14, r15, r23, r24, r25, r34, r35, r45])    # This part must be different
    rij = np.array([200, 100, 140, 30, 20, 40, 120, 30, 40, 10])

    # Construct the matrix A

    A = np.zeros((num_mics * (num_mics - 1) // 2, num_mics + 1))  # has form of 10 x 5
    row = 0

    # loop over microphone pairs
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            x_diff = 2 * (microphone_locations[j] - microphone_locations[i]).T

            A[row, 0] = x_diff[0]
            A[row, 1] = x_diff[1]
            A[row, j] = -2 * rij[row]

            # Assign zero to every column expect 0 and j
            for k in range(num_mics):
                if k != 0 and k != j:
                    A[row, k] = 0

            row += 1

    # Construct the matrix b

    b = np.zeros((num_mics * (num_mics - 1) // 2, 1))  # has form of 10 x 1
    row = 0
    # loop over microphone pairs
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            xi_norm_squarad = microphone_locations[i, 0] ** 2  # Extract the x-coordinates and normalize
            xj_norm_squarad = microphone_locations[j, 1] ** 2  # Extract the y-coordinates and normalize

            b[row] = rij[row] ** 2 - xi_norm_squarad - xj_norm_squarad

            row += 1

    return x_d

# # Compute the x and d
# x_d = solve_xandD(A, b)
# x = x_d[:2]  #select the first two elements from the arrary (x,y)
# d = x_d[2:]  #select from the third element to the end
#
# # Print the estimated location and nuisance parameters
# print("Estimated Location (x): ({:.2f}, {:.2f})".format(*[float(val) for val in x]))
# print("Estimated Nuisance Parameters (d):", d)
