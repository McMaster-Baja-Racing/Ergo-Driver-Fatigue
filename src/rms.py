import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data

# Root Mean Square (RMS) Calculation
# Relevance: https://svantek.com/applications/whole-body-vibration/#:~:text=ISO%202631,signal%20processor%20and%20a%20display

def calculate_rms(df, columns):
    """
    Compute the root-mean-square (RMS) value for the given columns.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing acceleration data.
        columns (list of str): List of column names for which to calculate RMS.
    
    Returns:
        dict: Dictionary with RMS values for each column.
    """
    rms_values = {}
    for col in columns:
        rms_values[col] = np.sqrt(np.mean(df[col]**2))
    return rms_values

def sliding_window_rms(series, window_size):
    """
    Compute the RMS over a sliding window.
    
    Parameters:
        series (pd.Series): The data series (e.g., acceleration values).
        window_size (int): Number of samples in the window.
    
    Returns:
        pd.Series: RMS computed for each window.
    """
    # Square the series
    squared = series ** 2
    # Compute the rolling mean of the squared values
    rolling_mean = squared.rolling(window=window_size, center=True).mean()
    # Take the square root to obtain the RMS
    return np.sqrt(rolling_mean)

# Example usage:
if __name__ == '__main__':
    file_path = "data/DATA_034.csv"
    
    processed_df = preprocess_data(
        file_path,
        static_orientation=True,
    )
    
    # Calculate the overall RMS for each weighted acceleration column
    columns = ['weighted_x', 'weighted_y', 'weighted_z', 'weighted_total']
    rms_values = calculate_rms(processed_df, columns)
    print("Overall RMS Values:")
    for col, val in rms_values.items():
        print(f"{col}: {val:.3f} m/s²")
    
    # Optionally, compute sliding window RMS (e.g., over a 1-second window)
    # Assuming your sample rate is around 100 Hz (adjust window_size as needed)
    window_size = 100  # 100 samples ~ 1 second if 100 Hz sampling rate
    processed_df['rms_total_window'] = sliding_window_rms(processed_df['weighted_total'], window_size)
    
    # Plot the sliding window RMS of the total acceleration
    plt.figure(figsize=(10, 6))
    plt.plot(processed_df['Timestamp'], processed_df['rms_total_window'], label='Sliding Window RMS (Total)')
    plt.xlabel("Time")
    plt.ylabel("RMS Acceleration (m/s²)")
    plt.title("Sliding Window RMS of Weighted Total Acceleration")
    plt.legend()
    plt.show()
