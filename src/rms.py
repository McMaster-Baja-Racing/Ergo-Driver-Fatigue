import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data

def calculate_rms(df, column):
    """
    Compute the root-mean-square (RMS) value for a given column.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing acceleration data.
        column (str): Column name for which to calculate RMS.
    
    Returns:
        float: RMS value.
    """
    return np.sqrt(np.mean(df[column]**2))

def sliding_window_rms(series, window_size):
    """
    Compute the RMS over a sliding window.
    
    Parameters:
        series (pd.Series): The data series (e.g., acceleration values).
        window_size (int): Number of samples in the window.
    
    Returns:
        pd.Series: RMS computed for each window.
    """
    squared = series ** 2
    rolling_mean = squared.rolling(window=window_size, center=True).mean()
    return np.sqrt(rolling_mean)

if __name__ == '__main__':
    # Paths to your two CSV files: one for the input and one for the seat measurement.
    file_path_input = "data/post_washers_with_ty_in_car/engine_rev_RUNNERS.CSV"
    file_path_seat = "data/post_washers_with_ty_in_car/engine_rev_SEAT.CSV"
    
    # Preprocess both datasets (make sure your preprocess_data function is set up to handle your data)
    # The preprocessed DataFrame is expected to include a datetime column and a 'weighted_total' column.
    input_df = preprocess_data(file_path_input, static_orientation=True)
    seat_df = preprocess_data(file_path_seat, static_orientation=True)
    
    # Calculate overall RMS for the weighted_total acceleration for each dataset.
    rms_input = calculate_rms(input_df, 'weighted_total')
    rms_seat  = calculate_rms(seat_df, 'weighted_total')
    
    print(f"Overall RMS Input: {rms_input:.3f} m/s²")
    print(f"Overall RMS Seat: {rms_seat:.3f} m/s²")
    
    # Compute the transmissibility ratio (or isolation index)
    transmissibility_ratio = rms_seat / rms_input
    print(f"Transmissibility Ratio: {transmissibility_ratio:.3f}")
    print(f"In decibels (dB): {20 * np.log10(transmissibility_ratio):.2f} dB")
    
    # Optionally, compute and plot the sliding window RMS to see how the isolation ratio changes over time.
    window_size = 100  # adjust based on your sampling rate (e.g., 100 samples ~ 1 sec at 100 Hz)
    input_df['rms_window'] = sliding_window_rms(input_df['weighted_total'], window_size)
    seat_df['rms_window'] = sliding_window_rms(seat_df['weighted_total'], window_size)
    
    # For comparison, we need to align the two datasets by time.
    # If the timestamps align, we can merge on the timestamp; otherwise, consider interpolation.
    # Here we perform an outer join on 'Timestamp' assuming both have a 'Timestamp' column.
    merged_df = pd.merge_asof(
        input_df.sort_values('Timestamp'),
        seat_df.sort_values('Timestamp'),
        on='Timestamp',
        suffixes=('_input', '_seat')
    )
    
    # Compute the sliding transmissibility ratio over time:
    merged_df['transmissibility_window'] = merged_df['rms_window_seat'] / merged_df['rms_window_input']
    
    # Plot the sliding window transmissibility ratio.
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['Timestamp'], merged_df['transmissibility_window'], label="Sliding Transmissibility Ratio")
    plt.xlabel("Time")
    plt.ylabel("Transmissibility Ratio")
    plt.title("Seat Transmissibility (Isolation) Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
