import pyaudio as audio
import numpy as np

#Find all audio devices visible to pyaudio
# pyaudio_handle = audio.PyAudio()
#
# for i in range(pyaudio_handle.get_device_count()):
#     device_info = pyaudio_handle.get_device_info_by_index(i)
#     print(i, device_info['name'])


Fs = 44100

# stream = pyaudio_handle.open(input_device_index=soundcard_naam, channels=5, format=pyaudio.paInt16, rate=Fs, input=True)

# samples = stream.read(N)
# data = np.frombuffer(samples, dtype='int16')

data_mic_0 = data[0::5]
data_mic_1 = data[1::5]
data_mic_2 = data[2::5]
data_mic_3 = data[3::5]
data_mic_4 = data[4::5]
