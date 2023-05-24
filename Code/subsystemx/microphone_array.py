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


def filtering(signal):
    Fpass_lower = 4000
    Fpass_higher = 8000
    Fstop_lower = 3000
    Fstop_higher = 9000
    pass_damp = 3
    stop_damp = 40

    # N, Wn = buttord(Fpass / Fs * 2, Fstop / Fs * 2, pass_damp, stop_damp)
    # b, a = butter(N, Fpass / Fs * 2)
    N, Wn = buttord([Fpass_lower / Fs * 2, Fpass_higher / Fs * 2], [Fstop_lower / Fs * 2, Fstop_higher / Fs * 2],
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


def peak(signal_reference, signal_recorded):
    # signal_reference = np.loadtxt(
    #     r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic3_reference_final.csv",
    #     delimiter=',')
    # signal_recorded = np.loadtxt(
    #     r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_middle_2_2.csv",
    #     delimiter=',')

    signal_recorded_filtered = filtering(signal_recorded)
    maxima, = sp.argrelmax(signal_recorded_filtered[1], order=800)
    if signal_recorded_filtered[1][maxima[0]] < 0.9 * max(signal_recorded_filtered[1]):
        maxima = maxima[1::]
    # print(maxima, maxima[1] - maxima[0])

    truncation = np.zeros((2, maxima[1] - maxima[0]))
    truncation[0] = signal_recorded[0][(maxima[0] - 130):(maxima[1] - 130)]
    truncation[1] = signal_recorded[1][(maxima[0] - 130):(maxima[1] - 130)]

    truncation_padded = np.zeros((2, math.ceil(max(truncation[0]))))
    truncation_padded[0] = np.linspace(0, len(truncation_padded[0]), len(truncation_padded[0]))
    truncation_padded[1] = np.concatenate((np.zeros(int(truncation[0][0])), truncation[1]))

    channel = ch3(signal_reference[1], truncation_padded[1], 0.01)

    maximum, = np.where(abs(channel) == max(abs(channel)))
    return maximum, channel


def tdoa_1(signal_reference_1, signal_recorded_1, signal_reference_2, signal_recorded_2):
    mic_1, _ = peak(signal_reference_1, signal_recorded_1)
    mic_2, _ = peak(signal_reference_2, signal_recorded_2)
    distance_mics = abs(mic_1 - mic_2)
    time = distance_mics / Fs
    distance = time * 34300
    print(distance_mics, distance)
    return distance


def isolation(recorded_signal, reference_signal):
    correlation = sp.correlate(recorded_signal[1], reference_signal[1], mode='same')
    # plt.plot(correlation)
    # plt.title("correlation")
    # plt.show()

    peak_index, = sp.argrelmax(correlation, order=800)
    # print("peak index = ", peak_index)
    if peak_index[0] < 1000:
        index = 1
    else:
        index = 0
    pulse_delay = peak_index[index] - (len(reference_signal[1]) // 2)

    isolated_pulse = np.zeros((2, len(reference_signal[0])))
    isolated_pulse[0] = recorded_signal[0][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]
    isolated_pulse[1] = recorded_signal[1][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]

    return isolated_pulse


def tdoa(signal_reference_1, signal_recorded_1, signal_reference_2, signal_recorded_2):
    signal_filtered_1 = filtering(signal_recorded_1)
    isolated_pulse_1 = isolation(signal_filtered_1, signal_reference_1)
    zeros = np.zeros(int(isolated_pulse_1[0][0]))
    channel_signal = np.zeros((2, math.ceil(max(isolated_pulse_1[0]))))
    channel_signal[0] = np.concatenate((zeros, isolated_pulse_1[0]))
    channel_signal[1] = np.concatenate((zeros, isolated_pulse_1[1]))

    signal_filtered_2 = filtering(signal_recorded_2)
    isolated_pulse_2 = isolation(signal_filtered_2, signal_reference_2)
    zeros = np.zeros(int(isolated_pulse_2[0][0]))
    channel_signal_2 = np.zeros((2, math.ceil(max(isolated_pulse_2[0]))))
    channel_signal_2[0] = np.concatenate((zeros, isolated_pulse_2[0]))
    channel_signal_2[1] = np.concatenate((zeros, isolated_pulse_2[1]))

    channel_1 = ch3(signal_reference_1[1], channel_signal[1], 0.01)
    channel_2 = ch3(signal_reference_2[1], channel_signal_2[1], 0.01)

    maximum_1, = np.where(abs(channel_1) == max(abs(channel_1)))
    maximum_2, = np.where(abs(channel_2) == max(abs(channel_2)))
    distance_mics = abs(maximum_1 - maximum_2)
    time = distance_mics / Fs
    distance = time * 34300
    print(distance_mics, distance)

# first recording
# 1-2 4.67
# 1-3 4.67
# 1-4 4.67
# 1-5 97.2
# 2-3 0
# 2-4 0
# 2-5 92.5
# 3-4 0
# 3-5 92.6
# 4-5 92.6

# third recording
# 1-2 10.11
# 1-3 3.8
# 1-4 8.56
# 1-5 96
# 2-3 0
# 2-4 1.56
# 2-5 92.6
# 3-4 13.22
# 3-5 92.6
# 4-5 637.8

# fourth recording
# 1-2 4.6
# 1-3 4.6
# 1-4 540.6
# 1-5 97.2
# 2-3 0
# 2-4 545.2
# 2-5 92.5
# 3-4 545.2
# 3-5 92.5
# 4-5 637.8

signal_reference_1 = np.loadtxt(
    r"E:\TU Delft\Github\EPO4\Code\References\mic1_reference_final.csv",
    delimiter=',')

# print(signal_reference_1.shape)

signal_recorded_1 = np.loadtxt(
    r"E:\TU Delft\Github\EPO4\Code\Square\Recording_170x160_1_1.csv",
    delimiter=',')

# plt.plot(signal_recorded_1[0], signal_recorded_1[1])
# plt.title("Recorded signal 1")
# plt.show()

signal_reference_2 = np.loadtxt(
    r"E:\TU Delft\Github\EPO4\Code\References\mic2_reference_final.csv",
    delimiter=',')

signal_recorded_2 = np.loadtxt(
    r"E:\TU Delft\Github\EPO4\Code\Square\Recording_170x160_1_2.csv",
    delimiter=',')


tdoa(signal_reference_1, signal_recorded_1, signal_reference_2, signal_recorded_2)
#
# isolated_pulse_1 = isolation(signal_recorded_1, signal_reference_1)
# # plt.plot(isolated_pulse_1[0], isolated_pulse_1[1])
# # plt.show()
#
# zeros = np.zeros(int(isolated_pulse_1[0][0]))
# channel_signal = np.zeros((2, math.ceil(max(isolated_pulse_1[0]))))
# channel_signal[0] = np.concatenate((zeros, isolated_pulse_1[0]))
# channel_signal[1] = np.concatenate((zeros, isolated_pulse_1[1]))
#
# signal_filtered = filtering(signal_recorded_2)
# isolated_pulse_2 = isolation(signal_filtered, signal_reference_2)
# # plt.plot(isolated_pulse_2[0], isolated_pulse_2[1])
# # plt.show()
#
# zeros = np.zeros(int(isolated_pulse_2[0][0]))
# channel_signal_2 = np.zeros((2, math.ceil(max(isolated_pulse_2[0]))))
# channel_signal_2[0] = np.concatenate((zeros, isolated_pulse_2[0]))
# channel_signal_2[1] = np.concatenate((zeros, isolated_pulse_2[1]))
#
# channel_1 = ch3(signal_reference_1[1], channel_signal[1], 0.01)
# channel_2 = ch3(signal_reference_2[1], channel_signal_2[1], 0.01)
#
# maximum_1, = np.where(abs(channel_1) == max(abs(channel_1)))
# maximum_2, = np.where(abs(channel_2) == max(abs(channel_2)))
# distance_mics = abs(maximum_1 - maximum_2)
# time = distance_mics / Fs
# distance = time * 34300
# print(distance_mics, distance)















# multilateration estimate_location (matrix)
# import numpy as np
# from scipy.linalg import pinv
#
# def estimate_location(microphone_locations, rij):
#     num_mics = microphone_locations.shape[0]  # number of microphones
#
#     # Construct the matrix A
#     A = np.zeros((num_mics * (num_mics - 1) // 2, num_mics + 1))  # create zero matrix with the correct shape
#     row = 0
#
#     # loop over microphone pairs
#     for i in range(num_mics):
#         for j in range(i + 1, num_mics):
#             x_diff = 2 * (microphone_locations[j] - microphone_locations[i]).T
#
#             A[row, 0] = x_diff[0]     #x-value
#             A[row, 1] = x_diff[1]     #y-value
#             A[row, j] = -2 * rij[row]    # range difference between microphones
#
#             # Assign zero to every column except 0 and j
#             for k in range(num_mics):
#                 if k != 0 and k != 1 and k != j:
#                     A[row, k] = 0
#
#             row += 1
#
#     # Construct the matrix b
#     b = np.zeros((num_mics * (num_mics - 1) // 2, 1))  # has form of 10 x 1
#     row = 0
#
#     # loop over microphone pairs
#     for i in range(num_mics):
#         for j in range(i + 1, num_mics):
#             xi_norm_squared = np.linalg.norm(microphone_locations[i]) ** 2  # Extract and normalize
#             xj_norm_squared = np.linalg.norm(microphone_locations[j]) ** 2
#
#             b[row] = rij[row] ** 2 - xi_norm_squared - xj_norm_squared
#
#             row += 1
#
#     # Solve for x and d
#     A_inv = pinv(A)  # pseudo-inverse
#     x_d = A_inv @ b
#     x = x_d[:2]  # select the first two elements from the array (x, y)
#     d = x_d[2:]  # select from the third element to the end
#
#     return x, d
#
# # Microphone locations
# microphone_locations = np.array([[0,480],[480,480],[480,0],[0,0],[0,240]])
#
# #Measured range differences (TDOA * speed of sound(343))
# # rij = np.array([r12, r13, r14, r15, r23, r24, r25, r34, r35, r45])    # This part needs to change
# rij = np.array([0.01*343, 0.02*343, 0.04*343, 0.05*343, 0.02*343, 0.03*343, 0.08*343, 0.02*343, 0.02*343, 0.1*343]) #random test values
#
# # Compute the x and d
# x, d = estimate_location(microphone_locations, rij)

# TODO: check the nuisance parameters, does it influence the estimate location?
# Print the estimated location and nuisance parameters
print("Estimated Location (x): ({:.2f}, {:.2f})".format(*[float(val) for val in x]))
print("Estimated Nuisance Parameters (d):", d)

# # Ideal OOK gold code plot
# # Set the parameters
# carrier_frequency = 6000  # 6 kHz
# bit_frequency = 2000  # 2kHz
# code = '11101011001110101001100101001111'  # "EB3A994F"
#
# # Calculate the time duration for each bit
# bit_duration = 1 / bit_frequency  # seconds
#
# # Calculate the time steps for generating the signal
# total_time = len(code) * bit_duration
# time = np.linspace(0, total_time, int(total_time * carrier_frequency))
#
# # Generate the carrier signal
# carrier_signal = np.sin(2 * np.pi * carrier_frequency * time)
#
# # Generate the OOK signal based on the code
# ook_signal = []
# for bit in code:
#     if bit == '1':
#         ook_signal.extend(np.ones(int(bit_duration * carrier_frequency)))
#     elif bit == '0':
#         ook_signal.extend(np.zeros(int(bit_duration * carrier_frequency)))
#
# # Plot the OOK signal
# plt.plot(time, ook_signal)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('OOK Signal')
# # plt.xlim([0, 0.03])
# plt.grid(True)
# plt.show()

# Ideal OOK refsignal
# from refsignal import refsignal
# from scipy.fft import fft, ifft
# Fs_TX = 44100
# Nbits = 64
# Timer0 = 1 # 10 kHz
# Timer1 = 8 # 5 kHz
# Timer3 = 2 # 3 Hz
# code = 0x92340f0faaaa4321 #1001001000110100000011110000111110101010101010100100001100100001#
#
# x, _ = refsignal(Nbits, Timer0, Timer1, Timer3, code, Fs_TX)
# X = fft(x)
#
# period = 1/Fs_TX
# t = np.linspace(0, len(x)*period, len(x))
# f = np.linspace(0, Fs_TX/1000, len(x))
#
# fig, ax = plt.subplots(2, 1, figsize=(10, 7))
#
# ax[0].plot(t, x)
# ax[0].set_title("Ideal OOK in the Time Domain")
# ax[0].set_xlabel("Time [s]")
# ax[0].set_ylabel("Magnitude")
# ax[0].set_xlim([0, 0.015])
# # ax[0].set_ylim([0, 2])
