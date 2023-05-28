from pyaudio import *
import numpy as np
from scipy.fft import fft, ifft
import scipy.signal as sp
import itertools

import robot

import pyaudio as audio

from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState
import math


class LocalizationSubSystem(subSystem):
    Fs = 44100
    pyaudioHandle = None
    deviceIndex = 1
    # durationRecording = 0.144
    durationRecording = 0.2
    i = 0

    def __init__(self):
        return

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.localizationState = self.state
        # self.pyaudioHandle = self.audio_devices(print_list=True)

        # set audio stuff on robot
        # robot.Robot.code = self.goldCode

    def update(self):
        if (self.state == subSystemState.Started) | (self.state == subSystemState.Running):
            self.state = subSystemState.Running
            robot.Robot.localizationState = self.state

        if not robot.Robot.speakerOn:
            robot.Robot.speakerOn = True
        # else:
        #     self.i += 1
        #     robot.Robot.speakerOn = False

        # robot.code = "EB3A994F"
        # robot.Robot.carrierFrequency = 6000
        # robot.Robot.bitFrequency = 2000
        # robot.Robot.repetitionCount = 64

        _mic_1, _mic_2, _mic_3, _mic_4, _mic_5 = self.microphone_array(self.deviceIndex, self.durationRecording)
        mics = self.microphone_array(self.deviceIndex, self.durationRecording)

        self.tdoa(_mic_1, _mic_2, _mic_3, _mic_4, _mic_5)
        # xy = self.localize(self.tdoa(_mic_1, _mic_2, _mic_3, _mic_4, _mic_5))

        # for j in range(1, 6):
        #     #     np.savetxt("Recording_reference_" + str(self.i) + "_" + str(j) + ".csv", mics[j], delimiter=",")
        #     np.savetxt(
        #         r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_137x162_test_3_" + str(j) + ".csv",
        #         mics[j - 1], delimiter=",")

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.localizationState = self.state

    def audio_devices(self, *, print_list: bool):
        """
        Find all audio devices visible to pyaudio, if print_list = True, print the list of audio_devices
        @param print_list: print the list of devices
        @return: returns the pyaudio handle
        """
        pyaudio_handle = PyAudio()

        if print_list:
            for i in range(pyaudio_handle.get_device_count()):
                device_info = pyaudio_handle.get_device_info_by_index(i)
                print(i, device_info['name'])

        return pyaudio_handle

    def microphone_array(self, _device_index, _duration_recording):
        """
        Records audio and splits it in 5 samples
        @param _device_index: Device index of the microphone controller. Depends on
        @param _duration_recording: length of the recording
        @return: returns samples of each of the 5 microphones
        """
        # Fs = 44100
        _number_of_samples = int(_duration_recording * self.Fs)  # np.round gebruiken

        _pyaudio_handle = self.audio_devices(print_list=False)
        _stream = _pyaudio_handle.open(input_device_index=_device_index, channels=5, format=audio.paInt16, rate=self.Fs,
                                       input=True)

        _samples = _stream.read(_number_of_samples)
        _data = np.frombuffer(_samples, dtype='int16')

        _data_length = len(_data[::5])
        _data_mic_1 = _data[0::5]
        _data_mic_2 = _data[1::5]
        _data_mic_3 = _data[2::5]
        _data_mic_4 = _data[3::5]
        _data_mic_5 = _data[4::5]

        _sample_axis_mic_1 = np.linspace(0, _data_length, _data_length)
        _sample_axis_mic_2 = np.linspace(0, _data_length, _data_length)
        _sample_axis_mic_3 = np.linspace(0, _data_length, _data_length)
        _sample_axis_mic_4 = np.linspace(0, _data_length, _data_length)
        _sample_axis_mic_5 = np.linspace(0, _data_length, _data_length)

        _mic_1 = _sample_axis_mic_1, _data_mic_1
        _mic_2 = _sample_axis_mic_2, _data_mic_2
        _mic_3 = _sample_axis_mic_3, _data_mic_3
        _mic_4 = _sample_axis_mic_4, _data_mic_4
        _mic_5 = _sample_axis_mic_5, _data_mic_5

        return _mic_1, _mic_2, _mic_3, _mic_4, _mic_5

    def ch3(self, y):
        epsi = 0.02
        signal_reference = self.reference_array()
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

    def reference_array(self):
        reference_mic = np.loadtxt(
            r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic1_reference_final.csv",
            delimiter=',')
        return reference_mic

    def isolation(self, recorded_signal):
        reference_signal = self.reference_array()
        correlation = sp.correlate(recorded_signal[1], reference_signal[1], mode='same')

        # peak_index, = sp.argrelmax(correlation, order=1000)
        # # print(peak_index)
        # if (peak_index[0] < 500):
        #     peak_index = np.delete(peak_index, [0], None)
        #
        # # print(peak_index)
        # if len(peak_index) == 2:
        #     pulse_delay = peak_index[0] - (len(reference_signal[1]) // 2)
        # else:
        #     pulse_delay = peak_index[1] - (len(reference_signal[1]) // 2)

        peak_index, = np.where(correlation == max(correlation))
        pulse_delay = int(peak_index - (len(reference_signal[1]) // 2))
        isolated_pulse = np.zeros((2, len(reference_signal[0])))
        isolated_pulse[0] = recorded_signal[0][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]
        isolated_pulse[1] = recorded_signal[1][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]
        return isolated_pulse


    def tdoa(self, signal_recorded_1, signal_recorded_2, signal_recorded_3, signal_recorded_4, signal_recorded_5):
        isolated_pulse_mic_1 = self.isolation(signal_recorded_1)
        zeros_1 = np.zeros(int(isolated_pulse_mic_1[0][0]))
        channel_signal_1 = np.zeros((2, math.ceil(isolated_pulse_mic_1[0][-1])))
        channel_signal_1[0] = np.concatenate((zeros_1, isolated_pulse_mic_1[0]))
        channel_signal_1[1] = np.concatenate((zeros_1, isolated_pulse_mic_1[1]))
        channel_1 = self.ch3(channel_signal_1[1])
        maximum_1, = np.where(abs(channel_1) == max(abs(channel_1)))

        isolated_pulse_mic_2 = self.isolation(signal_recorded_2)
        zeros_2 = np.zeros(int(isolated_pulse_mic_2[0][0]))
        channel_signal_2 = np.zeros((2, math.ceil(isolated_pulse_mic_2[0][-1])))
        channel_signal_2[0] = np.concatenate((zeros_2, isolated_pulse_mic_2[0]))
        channel_signal_2[1] = np.concatenate((zeros_2, isolated_pulse_mic_2[1]))
        channel_2 = self.ch3(channel_signal_2[1])
        maximum_2, = np.where(abs(channel_2) == max(abs(channel_2)))

        isolated_pulse_mic_3 = self.isolation(signal_recorded_3)
        zeros_3 = np.zeros(int(isolated_pulse_mic_3[0][0]))
        channel_signal_3 = np.zeros((2, math.ceil(isolated_pulse_mic_3[0][-1])))
        channel_signal_3[0] = np.concatenate((zeros_3, isolated_pulse_mic_3[0]))
        channel_signal_3[1] = np.concatenate((zeros_3, isolated_pulse_mic_3[1]))
        channel_3 = self.ch3(channel_signal_3[1])
        maximum_3, = np.where(abs(channel_3) == max(abs(channel_3)))

        isolated_pulse_mic_4 = self.isolation(signal_recorded_4)
        zeros_4 = np.zeros(int(isolated_pulse_mic_4[0][0]))
        channel_signal_4 = np.zeros((2, math.ceil(isolated_pulse_mic_4[0][-1])))
        channel_signal_4[0] = np.concatenate((zeros_4, isolated_pulse_mic_4[0]))
        channel_signal_4[1] = np.concatenate((zeros_4, isolated_pulse_mic_4[1]))
        channel_4 = self.ch3(channel_signal_4[1])
        maximum_4, = np.where(abs(channel_4) == max(abs(channel_4)))

        isolated_pulse_mic_5 = self.isolation(signal_recorded_5)
        zeros_5 = np.zeros(int(isolated_pulse_mic_5[0][0]))
        channel_signal_5 = np.zeros((2, math.ceil(isolated_pulse_mic_5[0][-1])))
        channel_signal_5[0] = np.concatenate((zeros_5, isolated_pulse_mic_5[0]))
        channel_signal_5[1] = np.concatenate((zeros_5, isolated_pulse_mic_5[1]))
        channel_5 = self.ch3(channel_signal_5[1])
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
            time[i] = distance[i] / self.Fs
            distance_cm[i] = time[i] * 34300
        print(distance_cm)
        return maximum

    def localize(self, maximum):
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
