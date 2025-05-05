import pandas as pd
import os
import glob
import re

def read_device_files(device_name, data_dir="../Data/raw"):
    """
    Reads all Excel files for a given device name, ignoring any prefix like date ranges.

    Args:
        device_name (str): device name, e.g. "8#Belt Conveyer"
        data_dir (str): Folder containing .xlsx files

    Returns:
        features_df (pd.DataFrame): Combined features with location column.
        rating_df (pd.DataFrame): Ratings.
    """
    # List all .xlsx files in the folder
    all_files = glob.glob(os.path.join(data_dir, "*.xlsx"))

    # Match files that contain the device name in parentheses
    pattern = f"\\({re.escape(device_name)}\\)"
    matched_files = [f for f in all_files if re.search(pattern, os.path.basename(f))]

    if not matched_files:
        raise FileNotFoundError(f"No files found for device: {device_name}")

    # Separate rating file from location files
    rating_file = None
    feature_files = []

    for f in matched_files:
        filename = os.path.basename(f)
        if re.search(r"\bRating\b", filename, re.IGNORECASE):
            rating_file = f
        else:
            feature_files.append(f)

    if not rating_file:
        raise FileNotFoundError(f"No Rating file found for device: {device_name}")

    # Read and combine all feature files
    feature_dfs = []
    for file in feature_files:
        filename = os.path.basename(file)
        location = filename.split(")")[-1].replace(".xlsx", "").strip()
        df = pd.read_excel(file)
        df["location"] = location
        feature_dfs.append(df)


    features_df = pd.concat(feature_dfs, ignore_index=True)
    rating_df = pd.read_excel(rating_file)

    return features_df, rating_df

if __name__ == "__main__":

    device = "8#Belt Conveyer"
    features_df, rating_df = read_device_files(device, data_dir="../Data/raw")

    features_df["datetime"] = pd.to_datetime(
        features_df["Date"].astype(str).str.strip() + " " +
        features_df["Time"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    rating_df["datetime"] = pd.to_datetime(
        rating_df["Date"].astype(str).str.strip() + " " +
        rating_df["Time"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    features_df.drop(columns=["Date", "Time", "id"], inplace=True)
    rating_df.drop(columns=["Date", "Time"], inplace=True)

    print("\n✅ Combined Feature Data:")
    print(features_df.head())
    print(features_df.shape)

    print("\n✅ Rating Data:")
    print(rating_df.head())
    
    pivot_features_df = features_df.pivot_table(
        index=["datetime", "location"],
        columns="Measurement",
        values="data"
    ).reset_index()

    pivot_rating_df = rating_df.pivot_table(
        index=["datetime", "Device"],
        columns="Metric",
        values="Rating"
    ).reset_index()

    pivot_features_df['Temperature'] = pivot_features_df.groupby('location')['Temperature'].ffill()

    print("\n✅ Pivoted Feature Data:")
    print(pivot_features_df.head())

    print("\n✅ Pivoted Rating Data:")
    print(pivot_rating_df.head())

    pivot_features_df = pivot_features_df.sort_values("datetime")
    pivot_rating_df = pivot_rating_df.sort_values("datetime")

    merged_df = pd.merge_asof(
        pivot_features_df,
        pivot_rating_df,
        on="datetime",
        direction="forward"
    )

    null_counts = merged_df.isna().sum()
    print("Null counts per column:")
    print(null_counts)

    output_path = os.path.join("../Data", "raw", f"{device}_merged.csv")
    merged_df.to_csv(output_path, index=False)



