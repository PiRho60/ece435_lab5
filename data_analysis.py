from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np


def read_csv(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read CSV file and return first two numeric columns, skipping bad/blank rows."""
    data = np.genfromtxt(
        file,
        delimiter=",",
        skip_header=2,   # adjust if your header length is different
        usecols=(0, 1),  # time + signal
    )

    # Handle case where only one row is returned
    if data.ndim == 1:
        data = data[None, :]

    # Drop rows with NaN/inf
    mask = np.isfinite(data).all(axis=1)
    data = data[mask]

    x_data = data[:, 0]  # time
    y_data = data[:, 1]  # signal
    return x_data, y_data


def compute_fft(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided FFT of y(t).

    Returns:
        freqs: frequency axis (Hz)
        mag:   magnitude spectrum |Y(f)|
    """
    N = len(y)
    # Estimate sampling interval from time axis
    dt = np.mean(np.diff(x))
    # Sampling frequency
    fs = 1.0 / dt

    # One-sided FFT (real input)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, d=dt)

    # Magnitude (you can normalize if you want)
    mag = np.abs(Y)

    return freqs, mag


def plot_time_and_fft(file: str):
    t, y = read_csv(file)
    freqs, mag = compute_fft(t, y)

    # --- Plot time-domain and frequency-domain in one figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Time domain
    ax1.plot(t, y)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time-domain Signal")
    ax1.grid(True)

    # Frequency domain
    ax2.plot(freqs, mag)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("|FFT|")
    ax2.set_title("Magnitude Spectrum")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()



def plot_fft(file: str):
    """Plot data from CSV file."""
    x_data, fft_data = read_csv(file)
    plt.plot(x_data, fft_data)
    plt.xlabel("X")
    plt.ylabel("FFT")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_time_and_fft("scope data lab 5/scope_1.csv")
    plot_fft("scope data lab 5/scope_2.csv")
