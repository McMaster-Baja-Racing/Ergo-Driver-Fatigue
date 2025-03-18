import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data
import matplotlib.pyplot as plt


def load_data_to_df(file_path):
    """
    Load the data from a CSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path)

# Function for fft
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
    filepath = "data/DATA_029.csv"
    processed_df = preprocess_data(filepath, static_orientation=True)

    # Run the FFT
    time_diffs = processed_df['Timestamp'].diff().dt.total_seconds().dropna()
    fs = 180
    print(f"Estimated sampling frequency: {fs:.2f} Hz")

    # Compute FFT for each axis
    for axis in ['Total Accel', 'Y', 'Z']:
        signal = processed_df[axis].dropna().values
        freqs, fft_magnitude = compute_fft(signal, fs)
        
        # Plot the FFT results
        plt.figure()
        plt.plot(freqs, fft_magnitude)
        plt.title(f'FFT of {axis}-axis')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()