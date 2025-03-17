import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_data

def compute_fft(signal, fs):
    """
    Compute the FFT of a signal and return positive frequencies and magnitudes.
    
    Parameters:
        signal (np.array): 1D array of signal values.
        fs (float): Sampling frequency in Hz.
    
    Returns:
        freqs (np.array): Frequencies (only positive frequencies).
        fft_magnitude (np.array): Magnitude of FFT corresponding to each frequency.
    """
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)
    
    # Take only the positive half of the spectrum
    pos_mask = freqs >= 0
    return freqs[pos_mask], np.abs(fft_vals[pos_mask])

if __name__ == '__main__':
    file_path = "data/DATA_034.csv"
    
    processed_df = preprocess_data(
        file_path,
        static_orientation=True,
    )

    # Estimate sampling frequency using timestamp differences (if using real data)
    time_diffs = processed_df['Timestamp'].diff().dt.total_seconds().dropna()
    fs = 1.0 / time_diffs.median()
    print(f"Estimated sampling frequency: {fs:.2f} Hz")
    
    # Compute the horizontal acceleration magnitude (combining x and y)
    processed_df['weighted_horizontal'] = np.sqrt(processed_df['weighted_x']**2 + processed_df['weighted_y']**2)
    
    # Compute FFT for vertical and horizontal acceleration components
    freqs_vertical, fft_mag_vertical = compute_fft(processed_df['weighted_z'].values, fs)
    freqs_horizontal, fft_mag_horizontal = compute_fft(processed_df['weighted_horizontal'].values, fs)
    
    # Plot vertical acceleration FFT
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_vertical, fft_mag_vertical, color='b')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Vertical Acceleration FFT")
    plt.xlim(0, 20)  # Focus on lower frequency range (commonly most relevant)
    plt.grid(True)
    plt.show()
    
    # Plot horizontal acceleration FFT
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_horizontal, fft_mag_horizontal, color='r')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Horizontal Acceleration FFT")
    plt.xlim(0, 20)  # Focus on lower frequencies; adjust as needed.
    plt.grid(True)
    plt.show()
