import logging
import pandas as pd
import os
import glob
import re
import argparse
import boto3
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "preprocessing.log")


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w"),
        ]
    )
    print(f"Logs are being stored in: {LOG_FILE}")
    logging.info(f"Log file location: {LOG_FILE}")


def log_dataframe_metadata(df, df_name):
    """
    This function provides a summary of key attributes of the DataFrame for debugging
    and monitoring purposes. It includes the shape of the DataFrame, its columns,
    the count of null values per column, memory usage, and a preview of the first few rows.

    Args:
        df (pd.DataFrame): The DataFrame to analyze and log metadata for.
        df_name (str): A descriptive name for the DataFrame to include in the logs (e.g., "Sensor DataFrame").

    Example Log:
        ============================================================================
        DATAFRAME SUMMARY: Train Dataset
        ============================================================================
        Shape: (10000, 12)
        Columns: col1, col2, col3, col4, ...
        ----------------------------------------------------------------------------
        Null values:
          col1                      : 0
          col2                      : 123
          col3                      : 0
          col4                      : 45
        ----------------------------------------------------------------------------
        Memory Usage: 1.56 MB
        ----------------------------------------------------------------------------
        Preview Dataframe Head:
          col1  col2  col3
          1      2     3
          4      5     6

    """
    separator = "=" * 80
    inner_separator = "-" * 80

    logging.info(f"\n{separator}")
    logging.info(f"DATAFRAME SUMMARY: {df_name}")
    logging.info(f"{separator}")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {', '.join(df.columns)}")
    logging.info(f"\n{inner_separator}")
    logging.info("Null values:")

    null_vals = df.isnull().sum()
    for col, val in null_vals.items():
        logging.info(f"  {col:25}: {val}")

    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info(f"\n{inner_separator}")
    logging.info(f"Memory Usage: {memory_usage:.2f} MB")

    logging.info(f"\n{inner_separator}")
    logging.info(f"Preview Dataframe Head:\n")
    preview = df.head(5).to_string(index=False).split('\n')
    for line in preview:
        logging.info(f"  {line}")

    logging.info(f"{separator}\n")


def read_files_from_s3(device_name, s3_bucket):
    """Read files from S3 bucket for given device"""
    logging.info(f"Connecting to AWS S3 bucket: {s3_bucket}")
    s3 = boto3.client('s3')
    # Always use the fixed S3 prefix
    prefix = 'raw/'
    all_files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.xlsx'):
                all_files.append(obj['Key'])
    logging.info(f"Files found in bucket: {all_files}")
    pattern = f"\\({re.escape(device_name)}\\)"
    matched_files = [f for f in all_files if re.search(pattern, os.path.basename(f))]
    if not matched_files:
        logging.error(f"No files found for device: {device_name} in S3 bucket {s3_bucket}")
        raise FileNotFoundError(f"No files found for device: {device_name} in S3 bucket {s3_bucket}")
    logging.info(f"Matched files for device {device_name}: {[os.path.basename(f) for f in matched_files]}")

    rating_files = []
    feature_files = []
    for f in matched_files:
        filename = os.path.basename(f)
        if re.search(r"\bRating\b", filename, re.IGNORECASE):
            rating_files.append(f)
        else:
            feature_files.append(f)
    if not rating_files:
        logging.error(f"No Rating file found for device: {device_name} in S3 bucket {s3_bucket}")
        raise FileNotFoundError(f"No Rating file found for device: {device_name} in S3 bucket {s3_bucket}")

    logging.info(f"Rating files: {rating_files}")
    logging.info(f"Feature files: {feature_files}")

    feature_dfs = []
    for file in feature_files:
        filename = os.path.basename(file)
        location = filename.split(")")[-1].replace(".xlsx", "").strip()
        obj = s3.get_object(Bucket=s3_bucket, Key=file)
        df = pd.read_excel(io.BytesIO(obj['Body'].read()))
        df["location"] = location
        feature_dfs.append(df)
    features_df = pd.concat(feature_dfs, ignore_index=True)

    rating_dfs = []
    for file in rating_files:
        obj = s3.get_object(Bucket=s3_bucket, Key=file)
        df = pd.read_excel(io.BytesIO(obj['Body'].read()))
        rating_dfs.append(df)
    rating_df = pd.concat(rating_dfs, ignore_index=True)

    return features_df, rating_df


def read_files_from_local(device_name, data_dir):
    """Read files from local directory for given device"""
    logging.info(f"Reading files from local directory: {data_dir}")
    # List all .xlsx files in the folder
    all_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    # Match files that contain the device name in parentheses
    logging.info(f"Files found in directory: {all_files}")
    pattern = f"\\({re.escape(device_name)}\\)"
    matched_files = [f for f in all_files if re.search(pattern, os.path.basename(f))]
    if not matched_files:
        logging.error(f"No files found for device: {device_name}")
        raise FileNotFoundError(f"No files found for device: {device_name}")
    logging.info(f"Matched files for device {device_name}: {[os.path.basename(f) for f in matched_files]}")

    # Separate rating files from location files
    rating_files = []
    feature_files = []
    for f in matched_files:
        filename = os.path.basename(f)
        if re.search(r"\bRating\b", filename, re.IGNORECASE):
            rating_files.append(f)
        else:
            feature_files.append(f)
    if not rating_files:
        logging.error(f"No Rating file found for device: {device_name}")
        raise FileNotFoundError(f"No Rating file found for device: {device_name}")

    logging.info(f"Rating files: {rating_files}")
    logging.info(f"Feature files: {feature_files}")

    feature_dfs = []
    for file in feature_files:
        filename = os.path.basename(file)
        location = filename.split(")")[-1].replace(".xlsx", "").strip()
        df = pd.read_excel(file)
        df["location"] = location
        feature_dfs.append(df)
    features_df = pd.concat(feature_dfs, ignore_index=True)

    rating_dfs = []
    for file in rating_files:
        df = pd.read_excel(file)
        rating_dfs.append(df)
    rating_df = pd.concat(rating_dfs, ignore_index=True)

    return features_df, rating_df


def read_device_files(device_name, data_dir="../data/raw", aws_mode=False, s3_bucket='brilliant-automation-capstone'):
    """
    Reads all Excel files for a given device name, ignoring any prefix like date ranges.
    If aws_mode is True, reads files from S3 bucket brilliant-automation-capstone/raw instead of local directory.
    """
    logging.info(f"Reading files for device: {device_name} (AWS mode: {aws_mode})")

    if aws_mode:
        features_df, rating_df = read_files_from_s3(device_name, s3_bucket)
    else:
        features_df, rating_df = read_files_from_local(device_name, data_dir)

    logging.info(f"features_df Date range: {features_df['Date'].min()} to {features_df['Date'].max()}")
    logging.info(f"rating_df Date range: {rating_df['Date'].min()} to {rating_df['Date'].max()}")
    return features_df, rating_df


def create_datetime_columns(features_df, rating_df):
    """Create datetime columns from Date and Time columns"""
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

    logging.info(f"features_df datetime range: {features_df['datetime'].min()} to {features_df['datetime'].max()}")
    logging.info(f"rating_df datetime range: {rating_df['datetime'].min()} to {rating_df['datetime'].max()}")

    return features_df, rating_df


def clean_dataframes(features_df, rating_df):
    """Drop unnecessary columns from dataframes"""
    features_df.drop(columns=["Date", "Time", "id"], inplace=True)
    rating_df.drop(columns=["Date", "Time"], inplace=True)
    return features_df, rating_df


def pivot_dataframes(features_df, rating_df):
    """Create pivot tables from the dataframes"""
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

    logging.info(
        f"pivot_features_df datetime range: {pivot_features_df['datetime'].min()} to {pivot_features_df['datetime'].max()}")
    logging.info(
        f"pivot_rating_df datetime range: {pivot_rating_df['datetime'].min()} to {pivot_rating_df['datetime'].max()}")

    return pivot_features_df, pivot_rating_df


def process_temperature_data(pivot_features_df):
    """Forward fill temperature data by location"""
    pivot_features_df['Temperature'] = pivot_features_df.groupby('location')['Temperature'].ffill()
    return pivot_features_df


def sort_dataframes(pivot_features_df, pivot_rating_df):
    """Sort dataframes by datetime"""
    pivot_features_df = pivot_features_df.sort_values("datetime")
    pivot_rating_df = pivot_rating_df.sort_values("datetime")
    return pivot_features_df, pivot_rating_df


def calculate_overlap_range(pivot_features_df, pivot_rating_df):
    """Calculate overlapping datetime range between feature and ratings data"""
    min_feature_time = pivot_features_df["datetime"].min()
    max_feature_time = pivot_features_df["datetime"].max()

    min_rating_time = pivot_rating_df["datetime"].min()
    max_rating_time = pivot_rating_df["datetime"].max()

    overlap_start = max(min_feature_time, min_rating_time)
    overlap_end = min(max_feature_time, max_rating_time)

    return overlap_start, overlap_end


def filter_datetime_range(df, start, end):
    """
    Filters a DataFrame to include rows where the values in the 'datetime' column
    fall within a given date range (inclusive).

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        start (datetime or str): The start of the date range (inclusive). Can be a datetime object or a valid date string.
        end (datetime or str): The end of the date range (inclusive). Can be a datetime object or a valid date string.

    Returns:
        pd.DataFrame: A filtered DataFrame containing rows where the values in the 'datetime' column
        are within the given date range.
    """
    logging.info(f"Filtering DataFrame by date range: {start} to {end} on the 'datetime' column")
    result_df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
    logging.info(f"Shape after filtering: {result_df.shape}")
    return result_df


def merge_dataframes(pivot_features_df, pivot_rating_df):
    """Merge feature and rating dataframes using merge_asof"""
    merged_df = pd.merge_asof(
        pivot_features_df,
        pivot_rating_df,
        on="datetime",
        direction="forward"
    )
    logging.info(f"merged_df datetime range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")
    logging.info("Merging completed.")
    return merged_df


def save_to_s3(merged_df, device, s3_bucket):
    """Save merged dataframe to S3"""
    s3 = boto3.client('s3')
    output_key = f"processed/{device}_merged.csv"
    csv_buffer = io.StringIO()
    merged_df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=s3_bucket, Key=output_key, Body=csv_buffer.getvalue())
    logging.info(f"Merged CSV written to s3://{s3_bucket}/{output_key}")


def save_to_local(merged_df, device, output_dir):
    """Save merged dataframe to local directory"""
    output_path = os.path.join(output_dir, f"{device}_merged.csv")
    merged_df.to_csv(output_path, index=False)
    logging.info(f"Merged CSV written to local path: {output_path}")


def save_merged_data(merged_df, device, output_dir, aws_mode, s3_bucket):
    """Save merged data to appropriate location"""
    if aws_mode:
        save_to_s3(merged_df, device, s3_bucket)
    else:
        save_to_local(merged_df, device, output_dir)


def process_device_data(device, data_dir, output_dir, aws_mode, s3_bucket):
    """
    Main processing function that orchestrates the entire data processing pipeline.

    Args:
        device (str): Device name
        data_dir (str): Directory containing raw data files
        output_dir (str): Directory to save preprocessed data
        aws_mode (bool): Whether to use AWS S3 or local files
        s3_bucket (str): S3 bucket name

    Returns:
        pd.DataFrame: The final merged dataframe
    """
    # Read device files
    features_df, rating_df = read_device_files(device, data_dir=data_dir, aws_mode=aws_mode, s3_bucket=s3_bucket)
    log_dataframe_metadata(features_df, "Sensor DataFrame")
    log_dataframe_metadata(rating_df, "Ratings DataFrame")

    # Create datetime columns
    features_df, rating_df = create_datetime_columns(features_df, rating_df)

    # Clean dataframes
    features_df, rating_df = clean_dataframes(features_df, rating_df)

    # Create pivot tables
    pivot_features_df, pivot_rating_df = pivot_dataframes(features_df, rating_df)

    # Process temperature data
    pivot_features_df = process_temperature_data(pivot_features_df)

    # Sort dataframes
    pivot_features_df, pivot_rating_df = sort_dataframes(pivot_features_df, pivot_rating_df)

    # Calculate overlap range and filter
    overlap_start, overlap_end = calculate_overlap_range(pivot_features_df, pivot_rating_df)
    pivot_features_df = filter_datetime_range(pivot_features_df, overlap_start, overlap_end)
    pivot_rating_df = filter_datetime_range(pivot_rating_df, overlap_start, overlap_end)

    # Merge dataframes
    merged_df = merge_dataframes(pivot_features_df, pivot_rating_df)
    log_dataframe_metadata(merged_df, "Merged DataFrame")

    # Save merged data
    save_merged_data(merged_df, device, output_dir, aws_mode, s3_bucket)

    return merged_df


def main():
    """Main function to handle command line arguments and execute processing"""
    parser = argparse.ArgumentParser(
        description="Preprocess sensor and ratings data for a device."
    )
    parser.add_argument("--device", required=True, help="Device name, e.g. '8#Belt Conveyer'")
    parser.add_argument("--data_dir", default="data/raw",
                        help="Directory containing raw .xlsx data files (default: data/raw)")
    parser.add_argument("--output_dir", default="data/preprocessed",
                        help="Directory to save preprocessed data (default: data/preprocessed)")
    parser.add_argument("--aws", action="store_true",
                        help="Read/write data from/to S3 bucket instead of local directory")

    args = parser.parse_args()
    device = args.device
    data_dir = args.data_dir
    output_dir = args.output_dir
    aws_mode = args.aws
    s3_bucket = 'brilliant-automation-capstone'

    # Setup logging
    setup_logging()

    logging.info("Starting preprocessing...")
    if not aws_mode:
        os.makedirs(output_dir, exist_ok=True)

    # Process device data
    merged_df = process_device_data(device, data_dir, output_dir, aws_mode, s3_bucket)

    return merged_df


if __name__ == "__main__":
    main()