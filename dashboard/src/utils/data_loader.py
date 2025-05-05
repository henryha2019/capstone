import pandas as pd
import os

DATA_DIR = "data"  # CSVs should be stored in this folder

def load_data(device, sensors):
    # Placeholder: In reality, you'd build the path using device
    file_path = os.path.join(DATA_DIR, f"{device}.csv")

    # Simulated dummy DataFrame (replace with pd.read_csv(file_path) later)
    df = pd.DataFrame({
        "sensor_location": ["Top", "Bottom", "Left", "Right"] * 25,
        "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="min"),
        "high_freq": range(100),
        "low_freq": [x * 0.5 for x in range(100)],
        "vibration": [x % 7 for x in range(100)],
        "temperature": [20 + (x % 5) for x in range(100)]
    })

    filtered_df = df[df["sensor_location"].isin(sensors)]
    return filtered_df
