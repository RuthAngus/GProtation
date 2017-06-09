from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz


def butter_bandpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a


def butter_bandpass_filter(data, lowcut, fs, order=5, plot=False):
    b, a = butter_bandpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    if plot:
        plot_bandpass(lowcut, fs, order)
    return y


def plot_bandpass(lowcut, fs, order):
    # Plot the frequency response for a few different orders.
    plt.clf()
    b, a = butter_bandpass(lowcut, fs, order=3)
    w, h = freqz(b, a, worN=5000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlim(0, 1)
    plt.savefig("butter_bandpass")


if __name__ == "__main__":

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)

    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')
    y = butter_bandpass_filter(x, lowcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.savefig("butter_filtered")
