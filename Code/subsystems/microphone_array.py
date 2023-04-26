<<<<<<< Updated upstream
stream = pyaudio_handle.open(input_device_index=device_index, channels=5, format=pyaudio.paInt16, rate=Fs, input=True)
=======
import pyaudio as audio

pyaudio_handle = audio.PyAudio()

for i in range(pyaudio_handle.get_device_count()):
    device_info = pyaudio_handle.get_device_info_by_index(i)
    print(i, device_info['name'])



Fs = 44100
# stream = pyaudio_handle.open(input_device_index= soundcard_naam, channels=5, format=pyaudio.paInt16, rate=Fs, input=True)

>>>>>>> Stashed changes
