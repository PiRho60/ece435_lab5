from typing import Any, Tuple
from matplotlib import pyplot as plt

import numpy as np


def read_csv(file: str) -> tuple[Any, Any]:
    """Read CSV file."""
    x_data, fft_data = np.loadtxt(file, delimiter=',', skiprows=2, unpack=True)
    return x_data, fft_data


def plot_data(file: str):
    """Plot data from CSV file."""
    x_data, fft_data = read_csv(file)
    plt.plot(x_data, fft_data)
    plt.show()

if __name__ == '__main__':
