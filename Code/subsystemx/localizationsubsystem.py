from pyaudio import *
import numpy as np
from scipy.fft import fft, ifft
import scipy.signal as sp

import robot
from subsystemx.subsystemStateEnum import subSystemState
from subsystemx.subsystem import subSystem

import pyaudio as audio

from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState
from scipy.signal import butter, buttord, lfilter
import math


class LocalizationSubSystem(subSystem):
    Fs = 44100
    pyaudioHandle = None
    deviceIndex = 1
    durationRecording = 0.096
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
        mics = self.microphone_array(self.deviceIndex, self.durationRecording)
        for j in range(1, 6):
            #     np.savetxt("Recording_reference_" + str(self.i) + "_" + str(j) + ".csv", mics[j], delimiter=",")
            np.savetxt(
                r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_mic4_bad_2_" + str(j) + ".csv",
                mics[j-1], delimiter=",")

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

    def peak(self, signal_reference, signal_recorded):
        # signal_reference = np.loadtxt(
        #     r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\References\mic3_reference_final.csv",
        #     delimiter=',')
        # signal_recorded = np.loadtxt(
        #     r"C:\Users\Djordi\OneDrive\Documents\Delft\Git\EPO4\Code\Square\Recording_middle_2_2.csv",
        #     delimiter=',')

        signal_recorded_filtered = self.filtering(signal_recorded)
        maxima, = sp.argrelmax(signal_recorded_filtered[1], order=800)

        truncation = np.zeros((2, maxima[1] - maxima[0]))
        truncation[0] = signal_recorded[0][(maxima[0] - 130):(maxima[1] - 130)]
        truncation[1] = signal_recorded[1][(maxima[0] - 130):(maxima[1] - 130)]

        truncation_padded = np.zeros((2, math.ceil(max(truncation[0]))))
        truncation_padded[0] = np.linspace(0, len(truncation_padded[0]), len(truncation_padded[0]))
        truncation_padded[1] = np.concatenate((np.zeros(int(truncation[0][0])), truncation[1]))

        channel = self.ch3(signal_reference[1], truncation_padded[1], 0.01)

        maximum, = np.where(abs(channel) == max(abs(channel)))
        return maximum
