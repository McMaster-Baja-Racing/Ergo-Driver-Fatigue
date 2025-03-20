import csv
from matplotlib import pyplot as plt


def plot_magnitude(ax, res, title_prefix=""):
    """
    Plot overall magnitude (time-domain) on the given Axes object (ax).
    """
    in_metrics = res['input_metrics']
    seat_metrics = res['seat_metrics']
    
    t_in = in_metrics['timestamps']
    t_seat = seat_metrics['timestamps']
    
    ax.plot(t_in, in_metrics['magnitude'], label="Runners")
    ax.plot(t_seat, seat_metrics['magnitude'], label="Seat")
    ax.set_title(f"{title_prefix} - Magnitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude")
    ax.legend()


def plot_sliding_rms(ax, res, title_prefix=""):
    """
    Plot sliding RMS on the given Axes object (ax).
    """
    in_metrics = res['input_metrics']
    seat_metrics = res['seat_metrics']
    
    t_in = in_metrics['timestamps']
    t_seat = seat_metrics['timestamps']
    
    ax.plot(t_in, in_metrics['sliding_rms'], label="Runners")
    ax.plot(t_seat, seat_metrics['sliding_rms'], label="Seat")
    ax.set_title(f"{title_prefix} - Sliding RMS")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS (m/s²)")
    ax.legend()


def plot_fft(ax, res, title_prefix=""):
    """
    Plot FFT (fft_total) on the given Axes object (ax).
    """
    in_metrics = res['input_metrics']
    seat_metrics = res['seat_metrics']
    
    freqs, _, _, _, fft_total_in = in_metrics['fft']
    _, _, _, _, fft_total_seat = seat_metrics['fft']
    
    ax.plot(freqs, fft_total_in, label="Runners")
    ax.plot(freqs, fft_total_seat, label="Seat")
    ax.set_title(f"{title_prefix} - FFT")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend()

    # Save fft_total_seat to a CSV file
    # csv_file_path = f"{title_prefix}_fft_total_seat.csv"
    # with open(csv_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Frequency (Hz)", "FFT Magnitude"])
    #     for f, mag in zip(freqs, fft_total_seat):
    #         writer.writerow([f, mag])
    # print(f"FFT data saved to {csv_file_path}")


def plot_sliding_transmissibility(ax, res, title_prefix=""):
    """
    Plot sliding transmissibility on the given Axes object (ax).
    """
    in_metrics = res['input_metrics']
    t_in = in_metrics['timestamps']
    
    ax.plot(t_in, res['sliding_ratio'], label="Sliding Transmissibility")
    ax.set_title(f"{title_prefix} - Sliding Transmissibility")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Transmissibility Ratio")
    ax.legend()

def plot_psd(ax, res, title_prefix=""):
    """
    Plot PSD on the given Axes object (ax).
    """
    in_metrics = res['input_metrics']
    seat_metrics = res['seat_metrics']
    
    freqs, psd_x_in, psd_y_in, psd_z_in, psd_combined_in = in_metrics['psd']
    _, psd_x_seat, psd_y_seat, psd_z_seat, psd_combined_seat = seat_metrics['psd']
    
    ax.plot(freqs, psd_combined_in, label="Runners")
    ax.plot(freqs, psd_combined_seat, label="Seat")
    ax.set_title(f"{title_prefix} - PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD ((m/s²)²/Hz)")
    ax.legend()

    # Save psd_combined_seat to a CSV file
    # csv_file_path = f"{title_prefix}_psd_combined_seat.csv"
    # with open(csv_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Frequency (Hz)", "PSD Magnitude"])
    #     for f, mag in zip(freqs, psd_combined_in):
    #         writer.writerow([f, mag])
    # print(f"PSD data saved to {csv_file_path}")


def plot_all_in_one(results_dict):
    """
    Create a gridded plot (N rows x 4 columns) where each row is a dataset
    and columns represent: 
      0: Overall Magnitude
      1: Sliding RMS
      2: FFT
      3: Sliding Transmissibility
    """
    dataset_keys = list(results_dict.keys())
    num_rows = len(dataset_keys)
    num_cols = 4

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), squeeze=False)
    
    for i, key in enumerate(dataset_keys):
        res = results_dict[key]
        
        plot_magnitude(axs[i][0], res, title_prefix=key)
        plot_sliding_rms(axs[i][1], res, title_prefix=key)
        plot_fft(axs[i][2], res, title_prefix=key)
        plot_sliding_transmissibility(axs[i][3], res, title_prefix=key)

    plt.tight_layout()
    plt.show()

def plot_single_dataset(results_dict, dataset_key, plot_type="fft"):
    """
    Plot a single dataset for a specific plot type: 
    'magnitude', 'sliding_rms', 'fft', or 'transmissibility'.
    """
    if dataset_key not in results_dict:
        print(f"Dataset '{dataset_key}' not found in results_dict.")
        return
    
    res = results_dict[dataset_key]
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    if plot_type == "magnitude":
        plot_magnitude(ax, res, title_prefix=dataset_key)
    elif plot_type == "sliding_rms":
        plot_sliding_rms(ax, res, title_prefix=dataset_key)
    elif plot_type == "fft":
        plot_fft(ax, res, title_prefix=dataset_key)
    elif plot_type == "transmissibility":
        plot_sliding_transmissibility(ax, res, title_prefix=dataset_key)
    elif plot_type == "psd":
        plot_psd(ax, res, title_prefix=dataset_key)
    else:
        print(f"Unknown plot_type: {plot_type}")
        return

    plt.tight_layout()
    plt.show()
