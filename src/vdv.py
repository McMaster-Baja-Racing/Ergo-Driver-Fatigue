import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_data

def compute_rms(series):
    """
    Compute the root-mean-square (RMS) value of a given series.
    
    Parameters:
        series (pd.Series or np.array): Acceleration values.
        
    Returns:
        float: The RMS value.
    """
    return np.sqrt(np.mean(series**2))

def compute_crest_factor(series):
    """
    Compute the crest factor, i.e., the ratio of the peak absolute acceleration
    to the RMS value.
    
    Parameters:
        series (pd.Series or np.array): Acceleration values.
        
    Returns:
        float: Crest factor.
    """
    rms_value = compute_rms(series)
    # Avoid division by zero
    if rms_value == 0:
        return np.nan
    return np.max(np.abs(series)) / rms_value

def compute_vdv(df, accel_col='weighted_total', timestamp_col='Timestamp'):
    """
    Compute the Vibration Dose Value (VDV) for a given acceleration column.
    
    VDV is defined as:
        VDV = (∫ a(t)^4 dt)^(1/4)
    For discrete data, we approximate the integral as a sum.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        accel_col (str): Column name for acceleration data (e.g., weighted_total).
        timestamp_col (str): Column name for the timestamp (assumed to be datetime).
        
    Returns:
        float: VDV in appropriate units (m/s^1.75).
    """
    # Calculate time differences in seconds.
    # We assume the timestamps are already converted to datetime.
    time_diff = df[timestamp_col].diff().dropna().dt.total_seconds().values
    if len(time_diff) == 0:
        raise ValueError("Insufficient timestamp data to compute dt.")
    # Assume nearly uniform sampling and take the median dt.
    dt = np.median(time_diff)
    
    # Compute the discrete integration: sum(a^4 * dt)
    integrated = np.sum(df[accel_col]**4 * dt)
    
    # VDV is the fourth root of the integrated value.
    vdv = integrated**(1/4)
    return vdv

# Example usage:
if __name__ == '__main__':
    file_path = "data/pre_washers_with_ty_in_car/engine_rev_SEAT_2.CSV"
    
    processed_df = preprocess_data(
        file_path,
        static_orientation=True,
    )
    
    # Calculate overall RMS for the weighted total acceleration.
    rms_total = compute_rms(processed_df['weighted_total'])
    crest_factor = compute_crest_factor(processed_df['weighted_total'])
    vdv = compute_vdv(processed_df, accel_col='weighted_total', timestamp_col='Timestamp')
    
    print(f"RMS (weighted_total): {rms_total:.3f} m/s²")
    print(f"Crest Factor: {crest_factor:.3f}")
    print(f"VDV: {vdv:.3f} m/s^(1.75)")
    
    # Plot the weighted_total acceleration and annotate the crest factor
    plt.figure(figsize=(10, 6))
    plt.plot(processed_df['Timestamp'], processed_df['weighted_total'], label='Weighted Total Acceleration')
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Weighted Total Acceleration")
    plt.legend()
    plt.show()
