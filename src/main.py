from vibration_data import VibrationData
import matplotlib.pyplot as plt
from helpful_functions import (
    compute_rms,
    compute_vdv,
    compute_fft,
    compute_crest_factor,
    compute_sliding_window_rms
)

def process_file(file_path, window_size=200):
    """
    Load vibration data from a CSV file, compute various metrics, and extract the data magnitude.
    
    Returns:
        A dictionary containing timestamps, computed metrics, FFT components, and overall data magnitude.
    """
    vib_data = VibrationData.from_csv(file_path=file_path)
    metrics = {
        'rms': compute_rms(vib_data),
        'vdv': compute_vdv(vib_data),
        'crest': compute_crest_factor(vib_data),
        'sliding_rms': compute_sliding_window_rms(vib_data, window_size),
        'fft': compute_fft(vib_data),
        'timestamps': vib_data.timestamps,
        'magnitude': vib_data.magnitude()  # Overall data magnitude
    }
    return metrics

def main():
    # Dictionary mapping labels to CSV file paths.
    file_dict_knocks = {
        "Pre Washer": "data/pre_washers_with_ty_in_car/whack_car_SEAT.CSV",
        "Post Washer": "data/post_washers_with_ty_in_car/whack_car_SEAT.CSV"
    }

    file_dict = {
        "Pre Washer": "data/pre_washers_with_ty_in_car/engine_rev_SEAT.CSV",
        "Post Washer": "data/post_washers_with_ty_in_car/engine_rev_SEAT.CSV"
    }
    
    results = {}
    
    # Process each file and print metrics.
    for label, path in file_dict.items():
        metrics = process_file(path)
        results[label] = metrics
        print(f"{label}:")
        print(f"  RMS: {metrics['rms']:.3f} m/s²")
        print(f"  VDV: {metrics['vdv']:.3f} m/s^(1.75)")
        print(f"  Crest Factor: {metrics['crest']:.3f}\n")

    # Plot overall data magnitude for all files.
    plt.figure(figsize=(10, 6))
    for label, metrics in results.items():
        plt.plot(metrics['timestamps'], metrics['magnitude'], label=f"{label} Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.title("Overall Data Magnitude")
    plt.legend()
    plt.show()
    
    # Plot Sliding Window RMS for all files.
    plt.figure(figsize=(10, 6))
    for label, metrics in results.items():
        plt.plot(metrics['timestamps'], metrics['sliding_rms'], label=f"{label} Sliding RMS")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS (m/s²)")
    plt.title("Sliding Window RMS of Vibration Data")
    plt.legend()
    plt.show()
    
    # Plot overall FFT for all files.
    plt.figure(figsize=(10, 6))
    for label, metrics in results.items():
        freqs, fft_x, fft_y, fft_z, fft_total = metrics['fft']
        plt.plot(freqs, fft_total, label=f"{label} FFT Total-axis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Overall FFT")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
