from pyaudio import *
import numpy as np
from scipy.fft import fft, ifft
import scipy.signal as sp
from scipy.linalg import pinv

import robot

import pyaudio as audio

from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState
from scipy.signal import butter, buttord, lfilter
import math


class LocalizationSubSystem(subSystem):
    Fs = 44100
    pyaudioHandle = None
    deviceIndex = 1
    durationRecording = 0.144
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
        # self.tdoa(_mic_1, _mic_2, _mic_3, _mic_4, _mic_5)
        self.estimate_location(self.tdoa_2(_mic_1, _mic_2, _mic_3, _mic_4, _mic_5))
        for j in range(1, 6):
            #     np.savetxt("Recording_reference_" + str(self.i) + "_" + str(j) + ".csv", mics[j], delimiter=",")
            np.savetxt(
                r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_middle_test_2_" + str(j) + ".csv",
                mics[j - 1], delimiter=",")

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

    def filtering(self, signal):
        _Fpass_lower = 4000
        _Fpass_higher = 7000
        _Fstop_lower = 3000
        _Fstop_higher = 8000
        _pass_damp = 3
        _stop_damp = 40

        # N, Wn = buttord(Fpass / Fs * 2, Fstop / Fs * 2, pass_damp, stop_damp)
        # b, a = butter(N, Fpass / Fs * 2)
        N, Wn = buttord([_Fpass_lower / self.Fs * 2, _Fpass_higher / self.Fs * 2],
                        [_Fstop_lower / self.Fs * 2, _Fstop_higher / self.Fs * 2],
                        _pass_damp, _stop_damp)
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

    def ch3(self, x, y, epsi):
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

    def reference_array(self):
        reference_mic_1 = np.loadtxt(
            r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic1_reference_final.csv",
            delimiter=',')

        reference_mic_2 = np.loadtxt(
            r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic2_reference_final.csv",
            delimiter=',')

        reference_mic_3 = np.loadtxt(
            r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic3_reference_final.csv",
            delimiter=',')

        reference_mic_4 = np.loadtxt(
            r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic4_reference_final.csv",
            delimiter=',')

        reference_mic_5 = np.loadtxt(
            r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic5_reference_final.csv",
            delimiter=',')

        reference_mics_array = [reference_mic_1, reference_mic_2, reference_mic_3, reference_mic_4, reference_mic_5]
        return reference_mics_array

    def isolation(self, recorded_signal, reference_signal):
        # reference_mics_array = self.reference_array()
        # mics_array = [_mic_1, _mic_2, _mic_3, _mic_4, _mic_5]
        # for i in range(5):
        #     correlation = sp.correlate(mics_array[i][1], reference_mics_array[i][1], mode='same')
        #     peak_index, = sp.argrelmax(correlation, order=800)
        #     if peak_index[0] < 1000:
        #         index = 1
        #     else:
        #         index = 0
        #     pulse_delay = peak_index[index] - (len(reference_mics_array[i][1]) // 2)
        #
        #     isolated_pulse = np.zeros((2, len(reference_mics_array[i][0])))
        #     isolated_pulse[0] = mics_array[i][0][pulse_delay:pulse_delay + len(mics_array[i][0] * 2)]
        #     isolated_pulse[1] = mics_array[i][1][pulse_delay:pulse_delay + len(mics_array[i][0] * 2)]
        #
        #     return isolated_pulse

        correlation = sp.correlate(recorded_signal[1], reference_signal[1], mode='same')

        peak_index, = sp.argrelmax(correlation, order=800)
        if peak_index[0] < 1000:
            index = 1
        else:
            index = 0
        pulse_delay = peak_index[index] - (len(reference_signal[1]) // 2)

        isolated_pulse = np.zeros((2, len(reference_signal[0])))
        isolated_pulse[0] = recorded_signal[0][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]
        isolated_pulse[1] = recorded_signal[1][pulse_delay:pulse_delay + len(reference_signal[0] * 2)]

        return isolated_pulse

    def tdoa(self, _mic_1, _mic_2, _mic_3, _mic_4, _mic_5):
        reference_mics_array = self.reference_array()

        signals_filtered_array = []
        signals_filtered_array.append(self.filtering(_mic_1))
        signals_filtered_array.append(self.filtering(_mic_2))
        signals_filtered_array.append(self.filtering(_mic_3))
        signals_filtered_array.append(self.filtering(_mic_4))
        signals_filtered_array.append(self.filtering(_mic_5))

        isolated_pulse_mic_1 = self.isolation(signals_filtered_array[0], reference_mics_array[0])
        zeros_1 = np.zeros(int(isolated_pulse_mic_1[0][0]))
        channel_signal_1 = np.zeros((2, math.ceil(max(isolated_pulse_mic_1[0]))))
        channel_signal_1[0] = np.concatenate((zeros_1, isolated_pulse_mic_1[0]))
        channel_signal_1[1] = np.concatenate((zeros_1, isolated_pulse_mic_1[1]))
        channel_1 = self.ch3(reference_mics_array[0][1], channel_signal_1[1], 0.01)
        maximum_1, = np.where(abs(channel_1) == max(abs(channel_1)))

        isolated_pulse_mic_2 = self.isolation(signals_filtered_array[1], reference_mics_array[1])
        zeros_2 = np.zeros(int(isolated_pulse_mic_2[0][0]))
        channel_signal_2 = np.zeros((2, math.ceil(max(isolated_pulse_mic_2[0]))))
        channel_signal_2[0] = np.concatenate((zeros_2, isolated_pulse_mic_2[0]))
        channel_signal_2[1] = np.concatenate((zeros_2, isolated_pulse_mic_2[1]))
        channel_2 = self.ch3(reference_mics_array[1][1], channel_signal_2[1], 0.01)
        maximum_2, = np.where(abs(channel_2) == max(abs(channel_2)))

        isolated_pulse_mic_3 = self.isolation(signals_filtered_array[2], reference_mics_array[2])
        zeros_3 = np.zeros(int(isolated_pulse_mic_3[0][0]))
        channel_signal_3 = np.zeros((2, math.ceil(max(isolated_pulse_mic_3[0]))))
        channel_signal_3[0] = np.concatenate((zeros_3, isolated_pulse_mic_3[0]))
        channel_signal_3[1] = np.concatenate((zeros_3, isolated_pulse_mic_3[1]))
        channel_3 = self.ch3(reference_mics_array[2][1], channel_signal_3[1], 0.01)
        maximum_3, = np.where(abs(channel_3) == max(abs(channel_3)))

        isolated_pulse_mic_4 = self.isolation(signals_filtered_array[3], reference_mics_array[3])
        zeros_4 = np.zeros(int(isolated_pulse_mic_4[0][0]))
        channel_signal_4 = np.zeros((2, math.ceil(max(isolated_pulse_mic_4[0]))))
        channel_signal_4[0] = np.concatenate((zeros_4, isolated_pulse_mic_4[0]))
        channel_signal_4[1] = np.concatenate((zeros_4, isolated_pulse_mic_4[1]))
        channel_4 = self.ch3(reference_mics_array[3][1], channel_signal_4[1], 0.01)
        maximum_4, = np.where(abs(channel_4) == max(abs(channel_4)))

        isolated_pulse_mic_5 = self.isolation(signals_filtered_array[4], reference_mics_array[4])
        zeros_5 = np.zeros(int(isolated_pulse_mic_5[0][0]))
        channel_signal_5 = np.zeros((2, math.ceil(max(isolated_pulse_mic_5[0]))))
        channel_signal_5[0] = np.concatenate((zeros_5, isolated_pulse_mic_5[0]))
        channel_signal_5[1] = np.concatenate((zeros_5, isolated_pulse_mic_5[1]))
        channel_5 = self.ch3(reference_mics_array[4][1], channel_signal_5[1], 0.01)
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

        time = np.zeros(10)
        distance_cm = np.zeros(10)
        for i in range(10):
            time[i] = distance[i] / self.Fs
            distance_cm[i] = time[i] * 34300
        print(distance_cm)
        return distance_cm

    def estimate_location(self, rij):
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
