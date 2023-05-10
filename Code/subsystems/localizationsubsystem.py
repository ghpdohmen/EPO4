import numpy as np
import robot
from subsystems.subsystemStateEnum import subSystemState
from subsystems.subsystem import subSystem
import pyaudio as audio


class LocalizationSubSystem(subSystem):
    Fs = 44100
    pyaudioHandle = None
    deviceIndex = 1
    durationRecording = 0.18181818181
    i = 0

    def __init__(self):
        return
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

        if robot.speakerOn == False:
            robot.speakerOn = True
        else:
            self.i += 1
            robot.speakerOn = False
        # robot.code = "EB3A994F"
        robot.carrierFrequency = 6000
        robot.bitFrequency = 2000
        robot.repetitionCount = 64
        mics = self.microphone_array(self.device_index, self.duration_recording)
        np.savetxt("Recording_reference_1_" + str(self.i) + ".csv", mics[0], delimiter=",")

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.localizationState = self.state



    def audio_devices(self,*, print_list: bool):
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

        _pyaudio_handle = self.audio_devices(print_list=False)
        _stream = _pyaudio_handle.open(input_device_index=_device_index, channels=5, format=audio.paInt16, rate=Fs,
                                     input=True)

        _samples = _stream.read(_number_of_samples)
        _data = np.frombuffer(_samples, dtype='int16')

        _data_length = len(_data[::5])
        _data_mic_0 = _data[0::5]
        _data_mic_1 = _data[1::5]
        _data_mic_2 = _data[2::5]
        _data_mic_3 = _data[3::5]
        _data_mic_4 = _data[4::5]

        _sample_axis_mic_0 = np.linspace(0, _data_length, _data_length)
        _sample_axis_mic_1 = np.linspace(0, _data_length, _data_length)
        _sample_axis_mic_2 = np.linspace(0, _data_length, _data_length)
        _sample_axis_mic_3 = np.linspace(0, _data_length, _data_length)
        _sample_axis_mic_4 = np.linspace(0, _data_length, _data_length)

        _mic_0 = _sample_axis_mic_0, _data_mic_0
        _mic_1 = _sample_axis_mic_1, _data_mic_1
        _mic_2 = _sample_axis_mic_2, _data_mic_2
        _mic_3 = _sample_axis_mic_3, _data_mic_3
        _mic_4 = _sample_axis_mic_4, _data_mic_4

        return _mic_0, _mic_1, _mic_2, _mic_3, _mic_4