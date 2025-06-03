import sys
import os
import argparse
import boto3
import io
import logging
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)

from utils import (
    load_and_clean_data,
    DEVICE_TARGET_FEATURES,
)

# Set up logging to console only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Map device names to their corresponding keys
DEVICE_MAP = {
    "8#Belt Conveyer": "conveyor_belt",
    "1#High-Temp Fan": "high_temp_fan",
    "Tube Mill": "tube_mill"
}

DEVICE_NAMES = ["8#Belt Conveyer", "1#High-Temp Fan", "Tube Mill"]

def process_device(device_name, aws_mode=False, s3_bucket='brilliant-automation-capstone'):
    """Process a single device's data, either from local files or S3."""
    logging.info(f"Processing data for {device_name} (AWS mode: {aws_mode})")
    device_key = DEVICE_MAP[device_name]
    target_features = DEVICE_TARGET_FEATURES[device_key]
    
    if aws_mode:
        input_key = f"process/{device_name}_merged.csv"
        output_key = f"process/{device_name}_full.csv"
        
        try:
            s3 = boto3.client('s3')
            # Read from S3
            obj = s3.get_object(Bucket=s3_bucket, Key=input_key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            
            # Process the data
            df = df.dropna()
            
            # Write back to S3
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=True)
            s3.put_object(Bucket=s3_bucket, Key=output_key, Body=csv_buffer.getvalue())
            logging.info(f"Successfully processed and saved data for {device_name} to s3://{s3_bucket}/{output_key}")
            
        except Exception as e:
            logging.error(f"Error processing {device_name} from S3: {str(e)}")
            raise
    else:
        DATA_DIR = f"../../Data/process/{device_name}_merged.csv"
        OUTPUT_DIR = f"../../Data/process/{device_name}_full.csv"
        
        try:
            df = load_and_clean_data(DATA_DIR)
            df.to_csv(OUTPUT_DIR, index=True)
            logging.info(f"Successfully processed and saved data for {device_name}")
        except Exception as e:
            logging.error(f"Error processing {device_name}: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and clean data for all devices."
    )
    parser.add_argument("--aws", action="store_true", help="Read/write data from/to S3 bucket instead of local directory")
    parser.add_argument("--s3-bucket", default="brilliant-automation-capstone", help="S3 bucket name (default: brilliant-automation-capstone)")
    
    args = parser.parse_args()
    
    logging.info("Starting data loading and cleaning...")
    
    for device_name in DEVICE_NAMES:
        try:
            process_device(device_name, aws_mode=args.aws, s3_bucket=args.s3_bucket)
        except Exception as e:
            logging.error(f"Failed to process {device_name}: {str(e)}")
            continue
    
    logging.info("Finished processing all devices")
