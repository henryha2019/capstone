import pandas as pd
from datetime import datetime
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from pathlib import Path
import threading
import boto3
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    aws_mode = False
    s3_bucket = 'brilliant-automation-capstone'
    s3_key = 'sample_belt_conveyer.csv'
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DataLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True
    
    def set_aws_mode(self, enabled: bool):
        self.aws_mode = enabled
        logger.info(f"AWS mode set to {enabled}")
        # Re-initialize to update data path
        self._initialize(force=True)
    
    def _initialize(self, force=False):
        """Initialize the data loader"""
        if hasattr(self, '_data_lock') and not force:
            return
        self._data_lock = threading.Lock()
        self._data = None
        self._last_update = None
        
        if self.aws_mode:
            logger.info(f"Data will be fetched from S3: s3://{self.s3_bucket}/{self.s3_key}")
        else:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            self._data_path = project_root / "Data" / "sample_belt_conveyer.csv"
            logger.info(f"Data path set to: {self._data_path}")
        
        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(self.update_data, 'interval', hours=1)
        self._scheduler.start()
        logger.info("DataLoader initialized with hourly updates")
        
    def _read_csv_from_s3(self):
        logger.info(f"Reading {self.s3_key} from S3 bucket {self.s3_bucket} into memory...")
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.s3_bucket, Key=self.s3_key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    
    def update_data(self):
        """Update the data from the CSV files"""
        try:
            logger.info("Starting data update...")
            if self.aws_mode:
                df = self._read_csv_from_s3()
            else:
                df = pd.read_csv(self._data_path)
            df = df.rename(columns={"datetime": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")
            
            # Only lock when updating the shared data
            with self._data_lock:
                self._data = df
                self._last_update = datetime.now()
            logger.info(f"Data updated successfully at {self._last_update}")
        except Exception as e:
            logger.error(f"Error updating data: {str(e)}")
            raise
    
    def get_data(self):
        """Get the current data"""
        if self._data is None:
            self.update_data()
        return self._data
    
    def get_last_update_time(self):
        """Get the timestamp of the last data update"""
        return self._last_update

# Create a global instance
data_loader = DataLoader()

def load_data():
    """Load data using the data loader"""
    return data_loader.get_data()

def get_unique_locations():
    """Get unique locations from the data"""
    df = data_loader.get_data()
    return sorted(df["location"].dropna().unique())

def get_unique_devices():
    """Get unique devices from the data"""
    df = data_loader.get_data()
    return sorted(df["Device"].unique())
