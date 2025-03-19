from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

@dataclass
class VibrationData:
    """
    A lightweight container for 3-axis accelerometer data along with timestamps
    and the sampling frequency.
    """
    timestamps: np.ndarray  # In seconds relative to the start
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    fs: float = field(default=None)  # Sampling frequency (Hz)
    
    @classmethod
    def from_csv(cls, file_path: str, 
                 time_col: str = "Timestamp", 
                 x_col: str = "X", 
                 y_col: str = "Y", 
                 z_col: str = "Z",
                 time_unit: str = "ms",
                 preprocess: bool = True,
                 cutoff: float = 0.3,
                 horizontal_weight: float = 1.4,
                 vertical_weight: float = 1.0):
        """
        Factory method to load accelerometer data from a CSV file.
        Timestamps are converted (default from milliseconds) to seconds relative to the start.
        If 'preprocess' is True, gravity is removed and weighting is applied.
        """
        df = pd.read_csv(file_path)
        df[time_col] = pd.to_datetime(df[time_col], unit=time_unit)
        df.sort_values(time_col, inplace=True)
        
        # Convert timestamps to seconds relative to the first sample.
        times = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().values
        x_data = df[x_col].values
        y_data = df[y_col].values
        z_data = df[z_col].values
        
        # Estimate sampling frequency
        dt = np.median(np.diff(times))
        fs = 1.0 / dt if dt > 0 else None
        
        instance = cls(timestamps=times, x=x_data, y=y_data, z=z_data, fs=fs)
        if preprocess:
            instance.preprocess(cutoff=cutoff,
                                horizontal_weight=horizontal_weight,
                                vertical_weight=vertical_weight)
        return instance
    
    def preprocess(self, cutoff: float = 0.3, 
                   horizontal_weight: float = 1.4, 
                   vertical_weight: float = 1.0):
        """
        Preprocess the data by removing gravity and applying weighting.
        Gravity is estimated by lowpass filtering each axis, and then subtracted.
        Weighting factors are applied to account for human sensitivity differences.
        """
        # Estimate gravity for each axis using a lowpass filter.
        gravity_x = lowpass_filter(self.x, cutoff, self.fs)
        gravity_y = lowpass_filter(self.y, cutoff, self.fs)
        gravity_z = lowpass_filter(self.z, cutoff, self.fs)
        
        # Remove gravity from each axis.
        self.x = self.x - gravity_x
        self.y = self.y - gravity_y
        self.z = self.z - gravity_z
        
        # Apply weighting factors.
        self.x *= horizontal_weight
        self.y *= horizontal_weight
        self.z *= vertical_weight
    
    def magnitude(self):
        """Return the overall acceleration magnitude as sqrt(x² + y² + z²)."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)


def butter_lowpass(cutoff: float, fs: float, order: int = 4):
    """
    Design a Butterworth lowpass filter.
    
    Parameters:
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.
        
    Returns:
        Filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4):
    """
    Apply a zero-phase Butterworth lowpass filter.
    
    Parameters:
        data: 1D array-like data to filter.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.
        
    Returns:
        Filtered data (gravity estimate).
    """
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)
