import random

import numpy as np

import robot
from subsystems.subsystem import subSystem
from subsystems.subsystemStateEnum import subSystemState
import pyaudio as audio


class LocalizationSubSystem(subSystem):
    Fs = 44100
    pyaudioHandle = None
    deviceIndex = 0
    durationRecording = 10
    bitStringLength = 200
    codeLength = 16
    goldCode = None

    def __init__(self):
        _poly1 = self.bit_string(self.bitStringLength)
        _poly2 = self.bit_string(self.bitStringLength)
        self.goldCode = self.gold_code(_poly1,_poly2,self.codeLength)

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.localizationState =  self.state
        self.pyaudioHandle = self.audio_devices(True)

        #set audio stuff on robot
        robot.Robot.code = self.goldCode
    def update(self):
        if (self.state == subSystemState.Started) | (self.state == subSystemState.Running):
            self.state = subSystemState.Running
            robot.Robot.localizationState = self.state
            print(self.goldCode)
    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.localizationState = self.state



    def audio_devices(*, print_list: bool):
        """
        Find all audio devices visible to pyaudio, if print_list = True, print the list of audio_devices
        @param print_list: print the list of devices
        @return: returns the pyaudio handle
        """
        pyaudio_handle = audio.PyAudio()

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
        _number_of_samples = _duration_recording * self.Fs
        N = _number_of_samples

        _pyaudio_handle = self.audio_devices(print_list=False)
        _stream = _pyaudio_handle.open(input_device_index=_device_index, channels=5, format=audio.paInt16, rate=Fs,
                                     input=True)

        _samples = _stream.read(N)
        _data = np.frombuffer(_samples, dtype='int16')

        _data_length = len(_data[::5])
        _data_mic_0 = _data[0::5]
        _data_mic_1 = _data[1::5]
        _data_mic_2 = _data[2::5]
        _data_mic_3 = _data[3::5]
        _data_mic_4 = _data[4::5]

        _sample_axis_mic_0 = np.linspace(0, _data_length / self.Fs, _data_length)
        _sample_axis_mic_1 = np.linspace(0, _data_length / self.Fs, _data_length)
        _sample_axis_mic_2 = np.linspace(0, _data_length / self.Fs, _data_length)
        _sample_axis_mic_3 = np.linspace(0, _data_length / self.Fs, _data_length)
        _sample_axis_mic_4 = np.linspace(0, _data_length / self.Fs, _data_length)

        _mic_0 = _sample_axis_mic_0, _data_mic_0
        _mic_1 = _sample_axis_mic_1, _data_mic_1
        _mic_2 = _sample_axis_mic_2, _data_mic_2
        _mic_3 = _sample_axis_mic_3, _data_mic_3
        _mic_4 = _sample_axis_mic_4, _data_mic_4

        return _mic_0, _mic_1, _mic_2, _mic_3, _mic_4

    def gold_code(self, _polynomial_1, _polynomial_2, _length):
        """

        @param _polynomial_1: First polynomial with random bit string
        @param _polynomial_2: Second polynomial with random bit string
        @param _length: length of the code
        @return: into a string
        """
        _poly1 = [int(c) for c in _polynomial_1]
        _poly2 = [int(c) for c in _polynomial_2]

        # convert polynomials to binary strings
        _poly1_str = ''.join(str(bit) for bit in _poly1)
        _poly2_str = ''.join(str(bit) for bit in _poly2)

        # set up LFSR registers
        _reg1 = int(_poly1_str, 2)
        _reg2 = int(_poly2_str, 2)

        _gold = []
        for i in range(_length):
            # XOR the outputs of the two registers
            _output = (_reg1 & 1) ^ (_reg2 & 1)
            _gold.append(_output)

            # shift the registers to the right by 1 bit
            _reg1 >>= 1
            _reg2 >>= 1

            # apply feedback to the registers
            _feedback1 = (_reg1 >> (len(_poly1) - 1)) ^ (_reg1 & 1)
            _feedback2 = (_reg2 >> (len(_poly2) - 1)) ^ (_reg2 & 1)
            _reg1 ^= _feedback1 << (len(_poly1) - 1)
            _reg2 ^= _feedback2 << (len(_poly2) - 1)

        # convert the list of bits to a binary string
        _gold_str = ''.join(str(_bit) for _bit in _gold)
        print(_gold_str)

        return _gold_str

    def bit_string(self, _length):
        """
        Generates a random bit string with a given length
        @param _length: length of the bit string
        @return: random bit string
        """
        _bit_string = ""
        for i in range(_length):
            _bit = random.randint(0, 1)
            _bit_string += str(_bit)

        return _bit_string



