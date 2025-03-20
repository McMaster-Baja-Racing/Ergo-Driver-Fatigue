import numpy as np
import pandas as pd
import scipy
from vibration_data import VibrationData

def compute_rms(vib_data: VibrationData) -> float:
    """
    Compute the overall root-mean-square (RMS) acceleration from the vibration data.
    
    Parameters:
        vib_data (VibrationData): An instance of VibrationData.
    
    Returns:
        float: The RMS acceleration.
    """
    mag = vib_data.magnitude()  # magnitude = sqrt(x² + y² + z²)
    return np.sqrt(np.mean(mag ** 2))

def compute_vdv(vib_data: VibrationData) -> float:
    """
    Compute the Vibration Dose Value (VDV) defined as:
        VDV = (∑ (a^4 * dt))^(1/4)
    where 'a' is the acceleration magnitude and dt is the sample interval.
    
    Parameters:
        vib_data (VibrationData): An instance of VibrationData.
    
    Returns:
        float: The VDV value.
    """
    dt = 1.0 / vib_data.fs
    mag = vib_data.magnitude()
    integrated = np.sum(mag ** 4 * dt)
    return integrated ** (1/4)

def compute_fft(vib_data: VibrationData) -> tuple:
    """
    Compute the FFT for each axis of the vibration data.
    
    Parameters:
        vib_data (VibrationData): An instance of VibrationData.
    
    Returns:
        tuple: (freqs, fft_x, fft_y, fft_z)
            - freqs (np.array): Positive frequency bins.
            - fft_x (np.array): Magnitude spectrum of x-axis.
            - fft_y (np.array): Magnitude spectrum of y-axis.
            - fft_z (np.array): Magnitude spectrum of z-axis.
    """
    n = len(vib_data.x)
    freqs = np.fft.fftfreq(n, d=1/vib_data.fs)
    fft_x = np.fft.fft(vib_data.x)
    fft_y = np.fft.fft(vib_data.y)
    fft_z = np.fft.fft(vib_data.z)
    fft_combined = np.sqrt(np.abs(fft_x)**2 + np.abs(fft_y)**2 + np.abs(fft_z)**2)
    pos_mask = freqs >= 0
    return freqs[pos_mask], np.abs(fft_x[pos_mask]), np.abs(fft_y[pos_mask]), np.abs(fft_z[pos_mask]), np.abs(fft_combined[pos_mask])

def compute_crest_factor(vib_data: VibrationData) -> float:
    """
    Compute the crest factor for the vibration data.
    Crest Factor is defined as the ratio of the peak absolute acceleration
    to the RMS value.
    
    Parameters:
        vib_data (VibrationData): An instance of VibrationData.
    
    Returns:
        float: Crest factor.
    """
    mag = vib_data.magnitude()
    rms_val = compute_rms(vib_data)
    if rms_val == 0:
        return np.nan
    return np.max(np.abs(mag)) / rms_val

def compute_sliding_window_rms(vib_data: VibrationData, window_size: int):
    """
    Compute the RMS over a sliding window for the vibration data magnitude.
    
    Parameters:
        vib_data (VibrationData): An instance of VibrationData.
        window_size (int): Number of samples in the sliding window.
        
    Returns:
        np.ndarray: Array of RMS values for each window.
    """
    # Convert magnitude to a pandas Series for rolling window computation.
    mag = vib_data.magnitude()
    series = pd.Series(mag)
    # Compute rolling RMS: first compute rolling mean of squares, then take sqrt.
    rolling_mean = series.pow(2).rolling(window=window_size, center=True).mean()
    return np.sqrt(rolling_mean).values


def compute_psd(vib_data: VibrationData, nperseg=256):
    """
    Compute the power spectral density (PSD) for each acceleration axis and the overall magnitude.
    
    Parameters:
        vib_data: An instance of VibrationData with attributes:
                  - timestamps (np.array): Time vector.
                  - x, y, z (np.array): Acceleration data for each axis.
                  - magnitude(): A method that returns the overall acceleration magnitude.
        nperseg (int): Number of samples per segment for Welch's method (default is 256).
    
    Returns:
        tuple: (freqs, psd_x, psd_y, psd_z, psd_magnitude)
            - freqs: Frequency values (Hz).
            - psd_x: PSD for the x-axis (units: (m/s²)²/Hz or g²/Hz if acceleration is in g).
            - psd_y: PSD for the y-axis.
            - psd_z: PSD for the z-axis.
            - psd_magnitude: PSD for the overall acceleration magnitude.
    """
    # Calculate sampling frequency from the timestamps.
    ts = vib_data.timestamps
    fs = 1 / np.mean(np.diff(ts))
    
    # Compute PSD for each acceleration axis using Welch's method.
    freqs, psd_x = scipy.signal.welch(vib_data.x, fs=fs, nperseg=nperseg)
    _, psd_y = scipy.signal.welch(vib_data.y, fs=fs, nperseg=nperseg)
    _, psd_z = scipy.signal.welch(vib_data.z, fs=fs, nperseg=nperseg)
    
    # Compute PSD for the overall magnitude.
    magnitude = vib_data.magnitude()
    _, psd_magnitude = scipy.signal.welch(magnitude, fs=fs, nperseg=nperseg)
    
    return freqs, psd_x, psd_y, psd_z, psd_magnitude