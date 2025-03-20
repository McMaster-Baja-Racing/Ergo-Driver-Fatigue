import numpy as np
import matplotlib.pyplot as plt
from vibration_data import VibrationData
from helpful_functions import (
    compute_rms,
    compute_vdv,
    compute_fft,
    compute_crest_factor,
    compute_sliding_window_rms
)

def process_file(file_path, window_size=200):
    """
    Load vibration data from a CSV file, compute various metrics, 
    and extract overall magnitude.
    
    Returns:
        dict: Contains timestamps, RMS, VDV, crest factor, sliding RMS,
              FFT components, and overall magnitude.
    """
    vib_data = VibrationData.from_csv(file_path=file_path)
    metrics = {
        'rms': compute_rms(vib_data),
        'vdv': compute_vdv(vib_data),
        'crest': compute_crest_factor(vib_data),
        'sliding_rms': compute_sliding_window_rms(vib_data, window_size),
        'fft': compute_fft(vib_data),  # returns (freqs, fft_x, fft_y, fft_z, fft_total)
        'timestamps': vib_data.timestamps,
        'magnitude': vib_data.magnitude()
    }
    return metrics

def process_transmissibility(input_file, seat_file, window_size=200):
    """
    Process the input and seat CSV files, compute metrics for both,
    and then calculate the transmissibility ratio.
    
    The overall transmissibility ratio is computed as the ratio of the 
    seat RMS to the input RMS, and its decibel value is also provided.
    The sliding transmissibility ratio is computed element-wise from the 
    sliding RMS values (assuming aligned timestamps).
    
    Returns:
        dict: Contains metrics for input and seat, overall transmissibility 
              ratio (and in dB), the sliding transmissibility ratio, and the 
              corresponding timestamps.
    """
    input_metrics = process_file(input_file, window_size)
    seat_metrics  = process_file(seat_file, window_size)
    
    overall_ratio = seat_metrics['rms'] / input_metrics['rms']
    overall_db = 20 * np.log10(overall_ratio)
    
    # Assuming the timestamps are aligned between input and seat data.
    sliding_ratio = seat_metrics['sliding_rms'] / input_metrics['sliding_rms']
    
    return {
        'input_metrics': input_metrics,
        'seat_metrics': seat_metrics,
        'overall_ratio': overall_ratio,
        'overall_db': overall_db,
        'sliding_ratio': sliding_ratio
    }

def main_combined():
    # Define your datasets as dictionaries.
    # Each dataset contains two accelerometer CSV paths:
    # "Runners" (e.g., input/runner) and "Seat" (e.g., seat or foam).
    datasets = {
        "Foam": {
            "Runners": "data/CarOn_Washer_Foam/Runner.csv",
            "Seat": "data/CarOn_Washer_Foam/Foam.csv"
        },
        "Washer": {
            "Runners": "data/CarOn_Tyler_Washer/Runner.csv",
            "Seat": "data/CarOn_Tyler_Washer/Seat.csv"
        },
        "No Washer": {
            "Runners": "data/CarOn_Tyler_NoWasher/Runner.csv",
            "Seat": "data/CarOn_Tyler_NoWasher/Seat.csv"
        }
    }
    
    # Process each dataset (i.e., each pair of CSV files) and store results.
    results_dict = {}
    window_size = 100  # adjust as needed based on your sampling rate
    for key, file_dict in datasets.items():
        res = process_transmissibility(file_dict["Runners"], file_dict["Seat"], window_size)
        results_dict[key] = res
        print(f"{key}:")
        print(f"  Input RMS: {res['input_metrics']['rms']:.3f} m/s²")
        print(f"  Seat RMS: {res['seat_metrics']['rms']:.3f} m/s²")
        print(f"  Transmissibility Ratio: {res['overall_ratio']:.3f}")
        print(f"  In decibels (dB): {res['overall_db']:.2f} dB\n")
    
    # Create a gridded plot.
    # Each row corresponds to one dataset and columns represent:
    # 0: Overall data magnitude, 1: Sliding window RMS,
    # 2: Overall FFT (using fft_total), 3: Sliding transmissibility ratio.
    num_rows = len(results_dict)
    num_cols = 4
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), squeeze=False)
    
    for i, (key, res) in enumerate(results_dict.items()):
        in_metrics = res['input_metrics']
        seat_metrics = res['seat_metrics']
        # Assume timestamps are the same for both channels.
        timestamps = in_metrics['timestamps']
        
        # --- Column 0: Overall Data Magnitude ---
        ax = axs[i][0]
        ax.plot(timestamps, in_metrics['magnitude'], label="Runners")
        ax.plot(seat_metrics['timestamps'], seat_metrics['magnitude'], label="Seat")
        ax.set_title(f"{key} - Magnitude")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnitude")
        ax.legend()
        
        # --- Column 1: Sliding Window RMS ---
        ax = axs[i][1]
        ax.plot(timestamps, in_metrics['sliding_rms'], label="Runners")
        ax.plot(seat_metrics['timestamps'], seat_metrics['sliding_rms'], label="Seat")
        ax.set_title(f"{key} - Sliding RMS")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RMS (m/s²)")
        ax.legend()
        
        # --- Column 2: Overall FFT ---
        # Using the overall FFT (fft_total) from each accelerometer.
        ax = axs[i][2]
        freqs, _, _, _, fft_total_in = in_metrics['fft']
        _, _, _, _, fft_total_seat = seat_metrics['fft']
        ax.plot(freqs, fft_total_in, label="Runners")
        ax.plot(freqs, fft_total_seat, label="Seat")
        ax.set_title(f"{key} - FFT")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.legend()
        
        # --- Column 3: Sliding Transmissibility Ratio ---
        ax = axs[i][3]
        ax.plot(timestamps, res['sliding_ratio'], label="Sliding Transmissibility")
        ax.set_title(f"{key} - Sliding Transmissibility")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Transmissibility Ratio")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_combined()
