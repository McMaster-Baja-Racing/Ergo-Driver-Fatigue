import pandas as pd

# Specify the input file name
input_file = 'data/CarOn_Washer_Foam/EngineRev_DoubleFoam_Runner-Seat.CSV'

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Get the unique sensor names (e.g., "Accelerometer 1", "Accelerometer 2")
sensors = df['Sensor'].unique()

# Process each sensor group separately
for sensor in sensors:
    # Filter rows for the current sensor
    sensor_df = df[df['Sensor'] == sensor].copy()
    
    # Remove the "Sensor" column
    sensor_df.drop(columns=['Sensor'], inplace=True)
    
    # Create a safe filename (replace spaces with underscores)
    output_file = sensor.replace(" ", "_") + ".csv"
    
    # Write the filtered DataFrame to a new CSV file
    sensor_df.to_csv(output_file, index=False)
    
    print(f"Data for {sensor} written to {output_file}")
