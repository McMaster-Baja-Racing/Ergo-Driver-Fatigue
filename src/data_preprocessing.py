import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- Preprocessing Functions ---

def butter_lowpass(cutoff, fs, order=4):
    """
    Design a Butterworth lowpass filter.
    
    Parameters:
        cutoff: Cutoff frequency (Hz).
        fs: Sampling frequency (Hz).
        order: Filter order.
        
    Returns:
        Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=4):
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
    y = filtfilt(b, a, data)
    return y

def rotation_matrix_from_vectors(a, b):
    """
    Compute the rotation matrix that rotates vector 'a' to align with vector 'b'
    using Rodrigues' rotation formula.
    
    Parameters:
        a: Source vector (as a NumPy array).
        b: Target vector (as a NumPy array).
    
    Returns:
        3x3 rotation matrix.
    """
    # Normalize the input vectors
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    # Compute the cross product and related parameters
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    
    # If vectors are already aligned, return the identity matrix
    if s < 1e-8:
        return np.eye(3)
    
    # Skew-symmetric cross-product matrix of v
    vx = np.array([[    0, -v[2],  v[1]],
                   [ v[2],     0, -v[0]],
                   [-v[1],  v[0],     0]])
    
    # Rodrigues' rotation formula
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
    return R

def preprocess_data(file_path,
                    timestamp_col='Timestamp',
                    x_col='X',
                    y_col='Y',
                    z_col='Z',
                    cutoff=0.3,
                    horizontal_weight=1.4,
                    vertical_weight=1.0,
                    static_orientation=True):
    """
    Load accelerometer data from a CSV, remove gravity, and rotate the data so that the
    gravity vector aligns with the z-axis. Finally, apply weighting factors.
    
    Parameters:
        file_path: Path to the CSV file.
        timestamp_col: Name of the timestamp column.
        x_col, y_col, z_col: Column names for the acceleration components.
        cutoff: Lowpass filter cutoff frequency (Hz) to estimate gravity.
        horizontal_weight: Weight to apply to horizontal (x, y) components.
        vertical_weight: Weight to apply to vertical (z) component.
        static_orientation: If True, compute one rotation matrix from the mean gravity vector.
        
    Returns:
        DataFrame with the original and processed (rotated, gravity-removed, weighted) data.
    """
    # Load data
    df = pd.read_csv(file_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
    df.sort_values(timestamp_col, inplace=True)
    
    # Estimate sampling frequency from the timestamp differences (assumes uniform sampling)
    time_diffs = df[timestamp_col].diff().dt.total_seconds().dropna()
    fs = 1.0 / time_diffs.median()
    print(f"File: {file_path} | Estimated sampling frequency: {fs:.2f} Hz")
    
    # Estimate gravity on each axis by applying a lowpass filter to the raw signals
    df['gravity_x'] = lowpass_filter(df[x_col], cutoff, fs)
    df['gravity_y'] = lowpass_filter(df[y_col], cutoff, fs)
    df['gravity_z'] = lowpass_filter(df[z_col], cutoff, fs)
    
    if static_orientation:
        # Compute the mean gravity vector from the lowpass filtered signals
        g_mean = np.array([df['gravity_x'].mean(), 
                           df['gravity_y'].mean(), 
                           df['gravity_z'].mean()])
        g_norm = np.linalg.norm(g_mean)
        if g_norm == 0:
            raise ValueError("Gravity vector norm is zero. Check your data or cutoff frequency.")
        g_mean_unit = g_mean / g_norm
        
        # Compute rotation matrix to rotate the mean gravity vector to align with [0, 0, 1]
        target = np.array([0, 0, 1])
        R = rotation_matrix_from_vectors(g_mean_unit, target)
        
        # Rotate the raw accelerometer data (each row is a 3D vector)
        raw_data = df[[x_col, y_col, z_col]].values
        rotated = np.dot(raw_data, R.T)
        df['rotated_x'] = rotated[:, 0]
        df['rotated_y'] = rotated[:, 1]
        df['rotated_z'] = rotated[:, 2]
        
        # (Optional) Rotate the gravity estimates to confirm alignment (should be mostly in z)
        gravity_data = df[['gravity_x', 'gravity_y', 'gravity_z']].values
        rotated_gravity = np.dot(gravity_data, R.T)
        df['rotated_gravity_x'] = rotated_gravity[:, 0]
        df['rotated_gravity_y'] = rotated_gravity[:, 1]
        df['rotated_gravity_z'] = rotated_gravity[:, 2]
        
        # Remove gravity in the rotated frame (gravity should be along z)
        df['linear_x'] = df['rotated_x']
        df['linear_y'] = df['rotated_y']
        df['linear_z'] = df['rotated_z'] - g_norm
        
    else:
        # Dynamic orientation (computing a rotation per sample)
        rotated_list = []
        for i, row in df.iterrows():
            g_vec = np.array([row['gravity_x'], row['gravity_y'], row['gravity_z']])
            g_norm = np.linalg.norm(g_vec)
            if g_norm == 0:
                R = np.eye(3)
            else:
                R = rotation_matrix_from_vectors(g_vec / g_norm, np.array([0, 0, 1]))
            raw_vec = np.array([row[x_col], row[y_col], row[z_col]])
            rotated_vec = R.dot(raw_vec)
            rotated_list.append(rotated_vec)
        rotated_array = np.array(rotated_list)
        df['rotated_x'] = rotated_array[:, 0]
        df['rotated_y'] = rotated_array[:, 1]
        df['rotated_z'] = rotated_array[:, 2]
        gravity_magnitudes = np.linalg.norm(df[['gravity_x', 'gravity_y', 'gravity_z']].values, axis=1)
        df['linear_x'] = df['rotated_x']
        df['linear_y'] = df['rotated_y']
        df['linear_z'] = df['rotated_z'] - gravity_magnitudes

    # Apply weighting factors to the linear (gravity-removed) acceleration
    df['weighted_x'] = df['linear_x'] * horizontal_weight
    df['weighted_y'] = df['linear_y'] * horizontal_weight
    df['weighted_z'] = df['linear_z'] * vertical_weight
    
    # Compute overall weighted acceleration magnitude
    df['weighted_total'] = np.sqrt(df['weighted_x']**2 + 
                                   df['weighted_y']**2 + 
                                   df['weighted_z']**2)
    
    return df

# --- Main Script for Plotting Both Input and Seat Data ---

if __name__ == '__main__':
    # Define file paths for input and seat measurements
    file_path_input = "data/pre_washers_with_ty_in_car/engine_rev_RUNNERS.CSV"
    file_path_seat  = "data/pre_washers_with_ty_in_car/engine_rev_SEAT.CSV"
    
    # Preprocess both datasets (adjust weighting factors if needed)
    input_df = preprocess_data(file_path_input,
                               static_orientation=True,
                               horizontal_weight=1,
                               vertical_weight=1)
    seat_df  = preprocess_data(file_path_seat,
                               static_orientation=True,
                               horizontal_weight=1,
                               vertical_weight=1)
    
    # --- Plot Comparison for Rotated Z (with gravity) ---
    plt.figure(figsize=(10, 6))
    plt.plot(input_df['Timestamp'], input_df['rotated_z'], label="Input Rotated Z (with gravity)")
    plt.plot(seat_df['Timestamp'], seat_df['rotated_z'], label="Seat Rotated Z (with gravity)")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Comparison of Rotated Z (with gravity)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Plot Comparison for Linear Z (gravity removed) ---
    plt.figure(figsize=(10, 6))
    plt.plot(input_df['Timestamp'], input_df['linear_z'], label="Input Linear Z (gravity removed)")
    plt.plot(seat_df['Timestamp'], seat_df['linear_z'], label="Seat Linear Z (gravity removed)")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Comparison of Linear Z (gravity removed)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Plot Comparison for Weighted Acceleration Components ---
    plt.figure(figsize=(10, 6))
    # For input data
    plt.plot(input_df['Timestamp'], input_df['weighted_x'], label="Input Weighted X")
    plt.plot(input_df['Timestamp'], input_df['weighted_y'], label="Input Weighted Y")
    plt.plot(input_df['Timestamp'], input_df['weighted_z'], label="Input Weighted Z")
    # For seat data (using a different linestyle or color to differentiate)
    plt.plot(seat_df['Timestamp'], seat_df['weighted_x'], label="Seat Weighted X", linestyle='--')
    plt.plot(seat_df['Timestamp'], seat_df['weighted_y'], label="Seat Weighted Y", linestyle='--')
    plt.plot(seat_df['Timestamp'], seat_df['weighted_z'], label="Seat Weighted Z", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Comparison of Weighted Acceleration Components")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Optionally, plot overall weighted_total for both ---
    plt.figure(figsize=(10, 6))
    plt.plot(input_df['Timestamp'], input_df['weighted_total'], label="Input Weighted Total")
    plt.plot(seat_df['Timestamp'], seat_df['weighted_total'], label="Seat Weighted Total", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Comparison of Weighted Total Acceleration")
    plt.legend()
    plt.grid(True)
    plt.show()
