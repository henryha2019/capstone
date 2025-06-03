import pandas as pd
from datetime import datetime
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from pathlib import Path
import threading
import boto3
import io
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    aws_mode = os.environ.get('DASHBOARD_AWS_MODE', '').lower() == 'true'
    s3_bucket = 'brilliant-automation-capstone'
    DEVICE_NAMES = [
        "8#Belt Conveyer",
        "1#High-Temp Fan",
        "Tube Mill"
    ]
    
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
                    logger.info(f"Initializing DataLoader with AWS mode from environment: {self.aws_mode}")
                    self._device_data = {}  # Dictionary to store data for each device
                    self._last_updates = {}  # Dictionary to store last update time for each device
                    self._initialize()
                    self._initialized = True
    
    def set_aws_mode(self, enabled: bool):
        self.aws_mode = enabled
        logger.info(f"AWS mode set to {enabled}")
        self._initialize(force=True)
    
    def _initialize(self, force=False):
        if hasattr(self, '_data_lock') and not force:
            return
        self._data_lock = threading.Lock()
        self._device_data = {}
        self._last_updates = {}
        
        if self.aws_mode:
            logger.info(f"Data will be fetched from S3 bucket: {self.s3_bucket}")
        else:
            current_file = Path(__file__)
            self._project_root = current_file.parent.parent.parent.parent
            logger.info(f"Project root set to: {self._project_root}")
        
        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(self.update_all_data, 'interval', hours=1)
        self._scheduler.start()
        logger.info("DataLoader initialized with hourly updates")
    
    def _get_s3_key(self, device_name):
        return f"process/{device_name}_full.csv"
    
    def _get_local_path(self, device_name):
        return self._project_root / "Data" / "process" / f"{device_name}_full.csv"
    
    def _read_csv_from_s3(self, device_name):
        s3_key = self._get_s3_key(device_name)
        logger.info(f"Reading {s3_key} from S3 bucket {self.s3_bucket} into memory...")
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    
    def update_all_data(self):
        """Update data for all devices"""
        for device_name in self.DEVICE_NAMES:
            try:
                self.update_data(device_name)
            except Exception as e:
                logger.error(f"Error updating data for {device_name}: {str(e)}")
    
    def update_data(self, device_name):
        """Update data for a specific device"""
        try:
            logger.info(f"Starting data update for {device_name}...")
            if self.aws_mode:
                df = self._read_csv_from_s3(device_name)
            else:
                df = pd.read_csv(self._get_local_path(device_name))
            
            df = df.rename(columns={"datetime": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")
            
            with self._data_lock:
                self._device_data[device_name] = df
                self._last_updates[device_name] = datetime.now()
            logger.info(f"Data updated successfully for {device_name} at {self._last_updates[device_name]}")
        except Exception as e:
            logger.error(f"Error updating data for {device_name}: {str(e)}")
            raise
    
    def get_data(self, device_name="8#Belt Conveyer"):
        """Get data for a specific device"""
        if device_name not in self._device_data:
            self.update_data(device_name)
        return self._device_data[device_name]
    
    def get_last_update_time(self, device_name="8#Belt Conveyer"):
        """Get last update time for a specific device"""
        return self._last_updates.get(device_name)

data_loader = DataLoader()

def load_data(device_name="8#Belt Conveyer", start_time=None, end_time=None):
    """Loads data for a specific device then filters and resample by user selected time range"""
    df = data_loader.get_data(device_name)
        
    if start_time and end_time and not df.empty:
        df = df[(df["timestamp"] >= pd.to_datetime(start_time)) &
                (df["timestamp"] <= pd.to_datetime(end_time))]

        delta = pd.to_datetime(end_time) - pd.to_datetime(start_time)
        if delta <= pd.Timedelta(hours=1):
            rule = None
        elif delta <= pd.Timedelta(days=1):
            rule = "5min"
        else:
            rule = "15min"

        if rule:
            df = (
                df.set_index("timestamp")
                .groupby(["location", "Device"])
                .resample(rule)
                .mean()
                .reset_index()
            )

    return df

def get_unique_locations(device_name="8#Belt Conveyer"):
    df = data_loader.get_data(device_name)
    return sorted(df["location"].dropna().unique())

def get_unique_devices():
    return list(data_loader.DEVICE_NAMES)
