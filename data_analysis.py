from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np


def read_csv(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read CSV file and return first two numeric columns, skipping bad/blank rows."""
    data = np.genfromtxt(
        file,
        delimiter=",",
        skip_header=2,   # your original skiprows
        usecols=(0, 1),  # only first two columns
    )

    # If there are blank lines / missing entries, they'll show up as NaN. Remove them.
    if data.ndim == 1:
        # In case there's only one row of data
        data = data[None, :]

    # Keep only rows where both columns are finite numbers
    mask = np.isfinite(data).all(axis=1)
    data = data[mask]

    x_data = data[:, 0]
    fft_data = data[:, 1]
    return x_data, fft_data


def plot_data(file: str):
    """Plot data from CSV file."""
    x_data, fft_data = read_csv(file)
    plt.plot(x_data, fft_data)
    plt.xlabel("X")
    plt.ylabel("FFT")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_data("scope data lab 5/scope_1.csv")
