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
from graphing import (
    plot_all_in_one,
    plot_single_dataset,
)
def process_file(file_path, window_size=200, simulate_isolation=False):

    vib_data = VibrationData.from_csv(file_path=file_path)
    
    # ------------------------------------------------
    # (1) Call our new "simulate_amplified_isolation" 
    #     to artificially damp seat vibrations:
    # ------------------------------------------------
    # You can adjust freq_center, freq_band, and attenuation_factor 
    # to see how the seat data might look with "stronger" isolation.
    if simulate_isolation:
        vib_data = simulate_amplified_isolation(
            vib_data,
            freq_center=10.0,
            freq_band=10.0,
            attenuation_factor=0.5
        )
        pass

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
    seat_metrics  = process_file(seat_file, window_size, simulate_isolation=False)
    
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

def simulate_amplified_isolation(vib_data, freq_center=10.0, freq_band=5.0, attenuation_factor=0.5):
    """
    Artificially reduce the seat vibration in a certain frequency band
    to mimic "stronger damping/isolation."

    :param vib_data: VibrationData object with x, y, z, timestamps, etc.
    :param freq_center: Center frequency (Hz) around which to apply extra damping.
    :param freq_band: +/- range around freq_center (Hz) for attenuation.
    :param attenuation_factor: Multiplier for amplitude in that band (e.g., 0.5 => 50% of original).
    :return: A new VibrationData object with modified x, y, z arrays.
    """
    import numpy as np
    import copy
    
    # Make a deep copy so we don't overwrite the original data.
    new_vib_data = copy.deepcopy(vib_data)

    # Retrieve time array and original accelerations
    t = vib_data.timestamps
    x_orig = vib_data.x
    y_orig = vib_data.y
    z_orig = vib_data.z

    # Estimate sampling frequency (assuming uniform spacing)
    dt = t[1] - t[0]
    fs = 1.0 / dt
    n = len(t)

    # Define the frequency array for the real FFT
    freqs = np.fft.rfftfreq(n, d=dt)

    def attenuate_in_band(signal):
        # 1) FFT
        fft_vals = np.fft.rfft(signal)

        # 2) Attenuate in the chosen frequency band
        f_low  = freq_center - freq_band
        f_high = freq_center + freq_band

        for i, f in enumerate(freqs):
            if f_low <= f <= f_high:
                fft_vals[i] *= attenuation_factor

        # 3) Inverse FFT
        return np.fft.irfft(fft_vals, n=n)

    # Apply to each axis
    new_vib_data.x = attenuate_in_band(x_orig)
    new_vib_data.y = attenuate_in_band(y_orig)
    new_vib_data.z = attenuate_in_band(z_orig)

    return new_vib_data


def main_combined():
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
    
    window_size = 100
    results_dict = {}
    for key, file_dict in datasets.items():
        res = process_transmissibility(file_dict["Runners"], file_dict["Seat"], window_size)
        results_dict[key] = res
        
        print(f"{key}:")
        print(f"  Input RMS: {res['input_metrics']['rms']:.3f} m/s²")
        print(f"  Seat RMS: {res['seat_metrics']['rms']:.3f} m/s²")
        print(f"  Transmissibility Ratio: {res['overall_ratio']:.3f}")
        print(f"  In decibels (dB): {res['overall_db']:.2f} dB\n")
    
    # plot_all_in_one(results_dict)
    plot_single_dataset(results_dict, "Foam", "sliding_rms")
    plot_single_dataset(results_dict, "Washer", "sliding_rms")
    plot_single_dataset(results_dict, "No Washer", "sliding_rms")
    

if __name__ == "__main__":
    main_combined()

