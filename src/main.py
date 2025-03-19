from vibration_data import VibrationData
import matplotlib.pyplot as plt
from helpful_functions import compute_rms, compute_vdv, compute_fft, compute_crest_factor, compute_sliding_window_rms


if __name__ == '__main__':
    filepath = "data/pre_washers_with_ty_in_car/engine_rev_SEAT.CSV"
    vibe_data = VibrationData.from_csv(filepath)

    # Plot the raw acceleration data
    plt.figure(figsize=(10, 6))
    plt.plot(vibe_data.timestamps, vibe_data.x, label='X', alpha=0.7)
    plt.plot(vibe_data.timestamps, vibe_data.y, label='Y', alpha=0.7)
    plt.plot(vibe_data.timestamps, vibe_data.z, label='Z', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.title("Raw Acceleration Data")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess the vibration data from a CSV.
    vib_data_pre_washer = VibrationData.from_csv(file_path="data/pre_washers_with_ty_in_car/engine_rev_SEAT.CSV")
    
    # Compute analysis metrics.
    rms_value = compute_rms(vib_data_pre_washer)
    vdv_value = compute_vdv(vib_data_pre_washer)
    crest = compute_crest_factor(vib_data_pre_washer)
    sliding_rms = compute_sliding_window_rms(vib_data_pre_washer, window_size=200)
    
    print(f"RMS: {rms_value:.3f} m/s²")
    print(f"VDV: {vdv_value:.3f} m/s^(1.75)")
    print(f"Crest Factor: {crest:.3f}")


    vib_data_post_washer = VibrationData.from_csv(file_path="data/post_washers_with_ty_in_car/engine_rev_SEAT.CSV")

    # Compute the RMS for the post-washer data.
    rms_post_washer = compute_rms(vib_data_post_washer)
    vdv_value_washer = compute_vdv(vib_data_post_washer)
    crest_post_washer = compute_crest_factor(vib_data_post_washer)
    sliding_rms_post_washer = compute_sliding_window_rms(vib_data_post_washer, window_size=200)

    # Plot the sliding window RMS.
    
    plt.figure(figsize=(10, 6))
    # Create a time vector for the sliding RMS (using the same timestamps as vib_data, trimmed if needed)
    time_vector = vib_data_pre_washer.timestamps
    time_vector_washer = vib_data_post_washer.timestamps
    plt.plot(time_vector, sliding_rms, label="Sliding Window RMS")
    plt.plot(time_vector_washer, sliding_rms_post_washer, label="Sliding Window RMS Post Washer")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS (m/s²)")
    plt.title("Sliding Window RMS of Vibration Data")
    plt.legend()
    plt.show()
    
    # Compute and plot the FFT for the Z-axis as an example.
    freqs, fft_x, fft_y, fft_z, fft_total = compute_fft(vib_data_pre_washer)
    freqs_washer, fft_x_washer, fft_y_washer, fft_z_washer, fft_total_washer = compute_fft(vib_data_post_washer)
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_total, label="FFT Total-axis")
    plt.plot(freqs_washer, fft_total_washer, label="FFT Total-axis Post Washer")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Overall FFT")
    plt.legend()
    plt.show()


