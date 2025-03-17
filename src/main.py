from data_preprocessing import preprocess_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Specify the path to your accelerometer CSV file
    file_path = "data/DATA_033.csv"
    
    # Preprocess the data with abstracted column names and static orientation assumption.
    processed_df = preprocess_data(
        file_path,
        static_orientation=False,  # Set to False if you want per-sample dynamic rotation
    )
    
    # Display the first few rows of the processed DataFrame
    print(processed_df.head())
    
    # Plot an example comparing the rotated z-axis before and after gravity removal
    plt.figure(figsize=(10, 6))
    plt.plot(processed_df['Timestamp'], processed_df['rotated_z'], label="Rotated Z (with gravity)")
    plt.plot(processed_df['Timestamp'], processed_df['linear_z'], label="Linear Z (gravity removed)")
    plt.plot(processed_df['Timestamp'], processed_df['Z'], label="Original Z", alpha=0.5, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Rotated vs. Gravity-Removed Z Acceleration")
    plt.legend()
    plt.show()

    # Plot an example comparing the rotated x-axis before and after gravity removal
    plt.figure(figsize=(10, 6))
    plt.plot(processed_df['Timestamp'], processed_df['rotated_x'], label="Rotated X (with gravity)")
    plt.plot(processed_df['Timestamp'], processed_df['linear_x'], label="Linear X (gravity removed)")
    plt.plot(processed_df['Timestamp'], processed_df['X'], label="Original X", alpha=0.5, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Rotated vs. Gravity-Removed X Acceleration")
    plt.legend()
    plt.show()

    # Plot now all three components of the weighted acceleration 
    plt.figure(figsize=(10, 6))
    plt.plot(processed_df['Timestamp'], processed_df['weighted_x'], label="Weighted X")
    plt.plot(processed_df['Timestamp'], processed_df['weighted_y'], label="Weighted Y")
    plt.plot(processed_df['Timestamp'], processed_df['weighted_z'], label="Weighted Z")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Weighted Acceleration Components")
    plt.legend()
    plt.show()
