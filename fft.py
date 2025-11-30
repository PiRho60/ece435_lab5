import numpy as np
import pandas as pd


def compute_snr_sfdr_from_fft(csv_path, skiprows=0):
    # Load CSV
    df = pd.read_csv(csv_path, skiprows=skiprows, header=None)

    # Convert both columns to numeric (handles +6.94E+00 correctly)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Column 1 = freq, Column 2 = magnitude
    freq = df.iloc[:, 0].values
    mag = df.iloc[:, 1].values

    # Drop NaN rows (in case weird header rows slip through)
    mask_valid = ~np.isnan(mag)
    mag = mag[mask_valid]

    # Fundamental = max magnitude
    fund_idx = np.argmax(mag)
    fund_power = mag[fund_idx] ** 2

    # Remove fundamental + harmonics
    mask = np.ones_like(mag, dtype=bool)
    for h in range(1, 6):
        idx = h * fund_idx
        if idx < len(mag):
            mask[max(0, idx - 5):min(len(mag), idx + 5)] = False

    noise_mag = mag[mask]
    noise_power = np.sum(noise_mag ** 2)

    # SNR
    SNR = 10 * np.log10(fund_power / noise_power)

    # SFDR
    spur_mag = noise_mag.max()
    SFDR = 20 * np.log10(mag[fund_idx] / spur_mag)

    return SNR, SFDR


SNR, SFDR = compute_snr_sfdr_from_fft("oscilloscope_fft_data/actual_GaussFT.csv", skiprows=3)
# SNR, SFDR = compute_snr_sfdr_from_fft("oscilloscope_fft_data/actual_GaussFT_10.csv", skiprows=3)
# SNR, SFDR = compute_snr_sfdr_from_fft("oscilloscope_fft_data/actual_GaussFT_15.csv", skiprows=3)
# SNR, SFDR = compute_snr_sfdr_from_fft("oscilloscope_fft_data/SinFT.csv", skiprows=3)
# SNR, SFDR = compute_snr_sfdr_from_fft("oscilloscope_fft_data/SinFT_1.csv", skiprows=3)
# SNR, SFDR = compute_snr_sfdr_from_fft("oscilloscope_fft_data/SinFT_2.csv", skiprows=3)
# SNR, SFDR = compute_snr_sfdr_from_fft("oscilloscope_fft_data/SqrFT.csv", skiprows=3)
print("SNR =", SNR, "dB")
print("SFDR =", SFDR, "dB")
