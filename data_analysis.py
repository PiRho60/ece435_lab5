from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import os


def read_csv(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read CSV file and return first two numeric columns, skipping bad/blank rows."""
    data = np.genfromtxt(
        file,
        delimiter=",",
        skip_header=2,   # adjust if your header length is different
        usecols=(0, 1),  # time + signal (or freq + mag for FFT files)
    )

    # Handle case where only one row is returned
    if data.ndim == 1:
        data = data[None, :]

    # Drop rows with NaN/inf
    mask = np.isfinite(data).all(axis=1)
    data = data[mask]

    x_data = data[:, 0]
    y_data = data[:, 1]
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
    ax1.set_title(f"Time-domain Signal\n{file}")
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
    """Plot data from CSV file (frequency vs amplitude)."""
    x_data, fft_data = read_csv(file)
    plt.figure()
    plt.plot(x_data, fft_data)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dBV)")
    plt.title(f"FFT: {os.path.basename(file)}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FFT_DIR = "oscilloscope_fft_data"

    # Loop over all CSV files in FFT_DIR and make a separate plot for each
    for fname in sorted(os.listdir(FFT_DIR)):
        if not fname.lower().endswith(".csv"):
            continue  # skip non-CSV files

        file_path = os.path.join(FFT_DIR, fname)
        print(f"Plotting {file_path}")
        plot_fft(file_path)
