import numpy as np
import matplotlib.pyplot as plt

Fs = 44100
Fmin = 0
Fmax = 12000
Duration = 5
# linear frequency chirp signal
def chirp_signal(Fmin, Fmax, Fs, Duration):

    N = Duration * Fs
    n = np.arange(0, N)
    f = np.linspace(Fmin, Fmax, N) / Fs
    x1 = np.sin(2 * np.pi * np.multiply(f, n))

    return x1

#generate chirp signal
x = chirp_signal(Fmin, Fmax, Fs, Duration)

#plot chirp signal
t = np.linspace(0, Duration, len(x))
plt.plot(t,x)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Linear Frequency Chirp Signal')
plt.show()

# summary:
# 1. sensitivity of the different frequencies using chirp_signal
# 2. Choose a carrier frequency within the available frequency band(suited to the sensitivity)
# 3. Bit frequency based on the specific data requirements
