import time

from pyaudio import *
import numpy as np
from scipy.fft import fft, ifft
# import scipy.signal as sp
import itertools
# import matplotlib.pyplot as plt

import robot

import pyaudio as audio

from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState
import math


class LocalizationSubSystem(subSystem):
    Fs = 44100
    pyaudioHandle = None
    deviceIndex = 1
    durationRecording = 0.15
    i = 0

    def __init__(self):
        self.array = []
        self.position_array = np.zeros(4)
        return

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.localizationState = self.state
        print("startpos:" + str(robot.Robot.startPos))
        self.position_array[0], self.position_array[1] = robot.Robot.startPos[0], robot.Robot.startPos[1]
        # self.pyaudioHandle = self.audio_devices(print_list=True)

    def update(self):
        if (self.state == subSystemState.Started) | (self.state == subSystemState.Running):
            self.state = subSystemState.Running
            robot.Robot.localizationState = self.state

        #Check if the robot is allowed to transmit pulses and allow it if it isn't
        if robot.Robot.speakerOn is False:
            robot.Robot.speakerOn = True

        # robot.code = "EB3A994F"
        # robot.Robot.carrierFrequency = 6000
        # robot.Robot.bitFrequency = 2000
        # robot.Robot.repetitionCount = 64

        # get the recordings for each microphone
        _mic_1, _mic_2, _mic_3, _mic_4, _mic_5 = self.microphone_array(self.deviceIndex, self.durationRecording)
        #
        previousTime = time.time_ns()
        #
        # signal_recorded_1 = self.tdoa_1(_mic_1)
        # signal_recorded_2 = self.tdoa_1(_mic_2)
        # signal_recorded_3 = self.tdoa_1(_mic_3)
        # signal_recorded_4 = self.tdoa_1(_mic_4)
        # signal_recorded_5 = self.tdoa_1(_mic_5)

        # print(signal_recorded_1, signal_recorded_2, signal_recorded_3, signal_recorded_4, signal_recorded_5)
        # distance = self.distance_calc(signal_recorded_1, signal_recorded_2, signal_recorded_3, signal_recorded_4, signal_recorded_5)
        # print(distance)
        # distance_cm = self.distance_cm(distance)

        # estimate location
        xy = self.estimate_location(self.tdoa(_mic_1, _mic_2, _mic_3, _mic_4, _mic_5))

        # xy = self.estimate_location(distance_calc(_mic_1, _mic_2, _mic_3, _mic_4, _mic_5))
        # print(xy)
        print(" estimate location delay: " + str((time.time_ns() - previousTime)/math.pow(10,9))) #todo: even kijken hoe lang dit is

        # Check whether the estimated location is within 75cm of the previous known location, if not, reuse the
        # previous value for estimation
        if ((xy[0] < 0 or xy[1] < 0) or (xy[0] > 480 or xy[1] > 480)):
            self.position_array[2], self.position_array[3] = self.position_array[0], self.position_array[1]
        else:
            self.position_array[2], self.position_array[3] = xy

        print("TDOA Location is: " + str(self.position_array[0]) + ", " + str( self.position_array[1]), '\n')
        # Send the location found in the previous update to the main file
        robot.Robot.posXLocalization = self.position_array[0]/100
        robot.Robot.posYLocalization = self.position_array[1]/100

        # Rewrite the location to be used in the next update to the values found in this update
        self.position_array[0], self.position_array[1] = self.position_array[2], self.position_array[3]
        self.position_array[2], self.position_array[3] = 0, 0

        # self.array.append(xy)
        # np.savetxt(
        #         r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_array_left_low.csv",
        #         self.array, delimiter=",")


        # print(self.position_array)

        # mics = self.microphone_array(self.deviceIndex, self.durationRecording)

        # plt.plot(self.array[0, 0::2], self.array[0, 1::2])
        # np.savetxt(
        #         r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_array.csv",
        #         self.array, delimiter=",")

        # for j in range(1, 6):
        #     np.savetxt(
        #         r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_236x122_test_1_" + str(j) + ".csv",
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
        """
        @param y: Recorded signal of 1 of the 5 microphones
        @return: returns the channel estimate as a 2-D array
        """
        # Set threshold parameter to 2%
        epsi = 0.02

        signal_reference = self.reference_array()  # Initialize known send signal x
        Ny = len(y)  # Length of y

        x = signal_reference[1]  # Initialize x to be the amplitude part of the known send signal

        # Deconvolution in frequency domain
        Y = fft(y)
        X = fft(x, Ny)
        H = Y / X

        # Threshold to avoid blow ups of noise during inversion
        # ii = np.absolute(X) < epsi * max(abs(X))
        # H[ii] = 0
        H[abs(X) < epsi * max(abs(X))] = 0

        h = np.real(ifft(H))  # ensure the result is real

        return h

    def reference_array(self):
        """
        @return: Returns the reference recording as a 2-D array
        """
        reference_mic = np.loadtxt(
            r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic1_reference_final.csv",
            delimiter=',')
        return reference_mic

    def estimate_location(self, distance):
        """
        @param distance: The distance of r12, r13, r14, r15, r23, r24, r25, r34, r35, and r45 as an 1-D array
        @return: Estimated x & y location of the robot as a 1-D array
        """
        coordinates_mics = np.array([[0, 480], [480, 480], [480, 0], [0, 0], [0, 240]])
        pairs = np.array(list(itertools.combinations([1, 2, 3, 4, 5], 2)))
        # Create indexes for all microphone pairs
        # pairs = list(
        #     itertools.combinations([1, 2, 3, 4, 5], 2))  # r12, r13, r14, r15, r23, r24, r25, r34, r35, r45

        A = np.zeros((10, 6))
        B = np.zeros((10, 1))

        first_column = np.zeros((10, 2))
        for row, [i, j] in enumerate(pairs):
            first_column[row] = (2 * (coordinates_mics[j - 1] - coordinates_mics[i - 1]))
            A[row, 0:2] = first_column[row]

            A[row, j] = -2 * distance[row]

            B[row] = pow(distance[row], 2) - pow(np.linalg.norm(coordinates_mics[i - 1]), 2) + pow(
                np.linalg.norm(coordinates_mics[j - 1]), 2)

        A_pinv = np.linalg.pinv(A)

        y = np.dot(A_pinv, B)

        xy = np.squeeze(y[0:2])
        # print(xy)
        return (xy)

    def tdoa(self, signal_recorded_1, signal_recorded_2, signal_recorded_3, signal_recorded_4, signal_recorded_5):
        """
        @param signal_recorded_1: The recording of microphone 1
        @param signal_recorded_2: The recording of microphone 2
        @param signal_recorded_3: The recording of microphone 3
        @param signal_recorded_4: The recording of microphone 4
        @param signal_recorded_5: The recording of microphone 5
        @return: The distance difference between r12, r13, r14, r15, r23, r24, r25, r34, r35, and r45 in cm as a 1-D array
        """

        channel_1 = self.ch3(signal_recorded_1[1])
        maximum_1, = np.where(abs(channel_1) == max(abs(channel_1)))

        channel_2 = self.ch3(signal_recorded_2[1])
        maximum_2, = np.where(abs(channel_2) == max(abs(channel_2)))

        channel_3 = self.ch3(signal_recorded_3[1])
        maximum_3, = np.where(abs(channel_3) == max(abs(channel_3)))

        channel_4 = self.ch3(signal_recorded_4[1])
        maximum_4, = np.where(abs(channel_4) == max(abs(channel_4)))

        channel_5 = self.ch3(signal_recorded_5[1])
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

        time = np.zeros(10)
        distance_cm = np.zeros(10)
        for i in range(10):
            time[i] = distance[i] / self.Fs
            distance_cm[i] = time[i] * 34300
        return distance_cm

    # def position_array(self, xy):
    #     position_array = np.zeros(4)
    #     position_array[0], position_array[1] = 0, 0
    #
    #     if self.update() == True:
    #         self.position_array[2], self.position_array[3] = xy
    #         # stuur deze hele array naar kalmann filter
    #
    #     self.position_array[0], self.position_array[1] = self.position_array[2], self.position_array[3]
    #     return

    # def plotter(self, xy):
    #     self.array.append(xy)
    #     array_plot = np.zeros((2, len(self.array) - 1))
    #     array_plot[0] = self.array[1::, 0]
    #     array_plot[1] = self.array[1::, 1]
    #     plt.plot(array_plot[0], array_plot[1])
    #     plt.show()

    def tdoa_1(self, signal_recorded):
        channel = self.ch3(signal_recorded[1])
        maximum, = np.where(abs(channel) == max(abs(channel)))
        return maximum

    def distance_calc(self, recorded_signal_1, recorded_signal_2, recorded_signal_3, recorded_signal_4, recorded_signal_5):
        distance = np.zeros(10)
        maximum_1 = self.tdoa_1(recorded_signal_1)
        maximum_2 = self.tdoa_1(recorded_signal_2)
        maximum_3 = self.tdoa_1(recorded_signal_3)
        maximum_4 = self.tdoa_1(recorded_signal_4)
        maximum_5 = self.tdoa_1(recorded_signal_5)

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
    
        return distance

    def distance_cm(self, distance):
        time = np.zeros(10)
        distance_cm = np.zeros(10)
        for i in range(10):
            time[i] = distance[i] / self.Fs
            distance_cm[i] = time[i] * 34300
        return distance_cm