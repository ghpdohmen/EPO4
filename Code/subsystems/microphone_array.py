import pyaudio as audio
import numpy as np
import matplotlib.pyplot as plt


# Find all audio devices visible to pyaudio, if print_list = True, print the list of audio_devices
def audio_devices(*, print_list: bool):
    pyaudio_handle = audio.PyAudio()

    if print_list:
        for i in range(pyaudio_handle.get_device_count()):
            device_info = pyaudio_handle.get_device_info_by_index(i)
            print(i, device_info['name'])

    return pyaudio_handle


# TODO: write this as a subsystem

def microphone_array(device_index, duration_recording):
    Fs = 44100
    number_of_samples = duration_recording * Fs
    N = number_of_samples * 5

    # pyaudio_handle = audio_devices(print_list=False)
    # stream = pyaudio_handle.open(input_device_index=device_index, channels=5, format=audio.paInt16, rate=Fs, input=True)
    #
    # samples = stream.read(N)
    # data = np.frombuffer(samples, dtype='int16')

    data = list(range(1, 101))
    data_length = len(data[::5])
    data_mic_0 = data[0::5]
    data_mic_1 = data[1::5]
    data_mic_2 = data[2::5]
    data_mic_3 = data[3::5]
    data_mic_4 = data[4::5]

    sample_axis_mic_0 = np.linspace(0, max(data_mic_0), data_length)
    sample_axis_mic_1 = np.linspace(0, max(data_mic_1), data_length)
    sample_axis_mic_2 = np.linspace(0, max(data_mic_2), data_length)
    sample_axis_mic_3 = np.linspace(0, max(data_mic_3), data_length)
    sample_axis_mic_4 = np.linspace(0, max(data_mic_4), data_length)

    mic_0 = sample_axis_mic_0, data_mic_0
    mic_1 = sample_axis_mic_1, data_mic_1
    mic_2 = sample_axis_mic_2, data_mic_2
    mic_3 = sample_axis_mic_3, data_mic_3
    mic_4 = sample_axis_mic_4, data_mic_4

    return mic_0, mic_1, mic_2, mic_3, mic_4


mics = microphone_array('Headset Earphone (HyperX Virtual Surround Sound)', 5)
for i in range(0, 5, 1):
    plt.plot(mics[i][0], mics[i][1])
    plt.show()
