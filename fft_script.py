import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# ==== User settings ====
TIME_FILE = "scope_data_lab_5/scope_10.csv"   # time-domain CSV
FREQ_FILE = "scope_data_lab_5/scope_12.csv"   # set to None if you don't have a freq-domain CSV

SAVE_COMPUTED_FFT_CSV = True
COMPUTED_FFT_CSV = "scope_data_lab_5/scope_trial_fft_from_python.csv"

# Any FFT bins below this level will be floored to this value (dBV)
DBV_FLOOR = -110.0


def read_time_csv(path: Path):
    """
    Read a Keysight-style time-domain CSV like scope_1.csv.
      row 0: column names  (e.g. 'x-axis', '1')
      row 1: units         (e.g. 'second', 'Volt')
      remaining rows: data
    """
    df = pd.read_csv(path, skiprows=[1])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    t = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    return t, y


def read_freq_csv(path: Path):
    """Read a Keysight-style frequency-domain CSV like scope_2.csv."""
    df = pd.read_csv(path, skiprows=[1])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    f = df.iloc[:, 0].to_numpy()
    dBV = df.iloc[:, 1].to_numpy()
    return f, dBV


def compute_fft(t, y):
    """
    Compute a one-sided FFT and return:
      freqs   : frequency axis [Hz]
      dBV     : magnitude in dBV (floored at DBV_FLOOR)
      fs      : sampling frequency [Hz]
    """
    # Remove DC offset
    y = y - np.mean(y)

    # Sampling interval and rate
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt_mean = np.mean(dt)
    fs = 1.0 / dt_mean
    N = len(y)

    # Real FFT (one-sided)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, d=dt_mean)

    # Amplitude spectrum (one-sided)
    mag = np.abs(Y) / N
    if mag.size > 2:
        mag[1:-1] *= 2.0

    # Peak amplitude → Vrms
    Vrms = mag / np.sqrt(2.0)

    # Floor everything below DBV_FLOOR
    vrms_floor = 10.0 ** (DBV_FLOOR / 20.0)  # Vrms corresponding to DBV_FLOOR
    Vrms = np.maximum(Vrms, vrms_floor)

    # Convert to dBV (0 dBV = 1 Vrms)
    dBV = 20.0 * np.log10(Vrms)

    return freqs, dBV, fs


def write_fft_csv(path: Path, freqs, dBV):
    """
    Write FFT result to a CSV with the same basic structure as scope_2.csv:
        row 0: 'x-axis', 'FFT'
        row 1: 'Hertz', 'dBV'
        rows 2+: data
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x-axis", "FFT"])
        writer.writerow(["Hertz", "dBV"])
        for f_val, mag in zip(freqs, dBV):
            writer.writerow([f_val, mag])


def main():
    # --- Load time-domain data ---
    time_path = Path(TIME_FILE)
    if not time_path.is_file():
        raise FileNotFoundError(f"Time-domain file not found: {time_path}")

    t, y = read_time_csv(time_path)

    # --- Compute FFT from time-domain data ---
    freqs, dBV, fs = compute_fft(t, y)
    f_nyq = fs / 2.0
    print(f"Loaded {len(t)} samples")
    print(f"Sampling frequency  fs ≈ {fs/1e6:.3f} MHz")
    print(f"Nyquist frequency   fs/2 ≈ {f_nyq/1e6:.3f} MHz")
    print("Note: the FFT from scope_1.csv is only valid up to fs/2.")

    # --- Optionally load existing scope FFT for comparison ---
    have_scope_fft = False
    f_scope = dBV_scope = None
    if FREQ_FILE is not None:
        freq_path = Path(FREQ_FILE)
        if freq_path.is_file():
            f_scope, dBV_scope = read_freq_csv(freq_path)
            have_scope_fft = True
        else:
            print(f"Warning: FREQ_FILE '{FREQ_FILE}' not found. "
                  "Continuing without scope FFT comparison.")

    # --- Optionally save computed FFT to a CSV ---
    if SAVE_COMPUTED_FFT_CSV:
        out_path = Path(COMPUTED_FFT_CSV)
        write_fft_csv(out_path, freqs, dBV)
        print(f"Computed FFT saved to: {out_path}")

    # --- Plot time-domain and frequency-domain ---
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 8))

    # Time domain (µs for readability)
    ax_time.plot(t * 1e6, y)
    ax_time.set_title("Time Domain Signal")
    ax_time.set_xlabel("Time [µs]")
    ax_time.set_ylabel("Amplitude [V]")
    ax_time.grid(True)

    # Frequency domain
    f_python_mhz = freqs / 1e6
    ax_freq.plot(f_python_mhz, dBV, label="FFT from Python")

    if have_scope_fft:
        ax_freq.plot(f_scope / 1e6, dBV_scope,
                     linestyle="--", alpha=0.7,
                     label="FFT from scope_2.csv")

    ax_freq.set_title("Frequency Domain (Magnitude Spectrum)")
    ax_freq.set_xlabel("Frequency [MHz]")
    ax_freq.set_ylabel("Magnitude [dBV]")
    ax_freq.grid(True)
    if have_scope_fft:
        ax_freq.legend()

    # X-axis: cover all data (Python + scope_2 if present)
    x_max = np.max(f_python_mhz)
    # x_max = 40
    # if have_scope_fft:
    #     x_max = max(x_max, np.max(f_scope / 1e6))
    ax_freq.set_xlim(0, x_max)

    # Y-axis: floor at DBV_FLOOR, top just above the max peak
    y_max = np.max(dBV)
    if have_scope_fft:
        y_max = max(y_max, np.max(dBV_scope))
    ax_freq.set_ylim(DBV_FLOOR, y_max + 5.0)  # +5 dB headroom

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
