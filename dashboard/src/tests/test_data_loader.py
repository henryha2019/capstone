import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import io
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.data_loader import DataLoader, load_data, get_unique_locations, get_unique_devices


@pytest.fixture
def sample_csv_data():
    return pd.DataFrame({
        'datetime': pd.date_range('2023-01-01 10:00:00', periods=100, freq='10min'),
        'Device': ['8#Belt Conveyer'] * 100,
        'location': ['Motor Drive End', 'Motor Non-Drive End', 'Gear Reducer'] * 33 + ['Motor Drive End'],
        'Temperature': np.random.normal(25, 3, 100),
        'High-Frequency Acceleration': np.random.normal(10, 2, 100),
        'Vibration Velocity Z': np.random.normal(2, 0.5, 100),
        'velocity_rms': np.random.choice([0, 1, 2], 100),
        'crest_factor': np.random.choice([0, 1, 2], 100),
        'Unnamed: 0': range(100)
    })


@pytest.fixture
def sample_high_temp_fan_data():
    return pd.DataFrame({
        'datetime': pd.date_range('2023-01-01 10:00:00', periods=50, freq='10min'),
        'Device': ['1#High-temperature Fan'] * 50,
        'location': ['Free End'] * 50,
        'Temperature': np.random.normal(30, 2, 50)
    })


class TestDataLoader:
    
    def test_singleton_pattern(self):
        loader1 = DataLoader()
        loader2 = DataLoader()
        assert loader1 is loader2
    
    @patch('boto3.client')
    def test_s3_loading_basic(self, mock_boto3, sample_csv_data):
        csv_buffer = io.StringIO()
        sample_csv_data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        mock_s3 = Mock()
        mock_boto3.return_value = mock_s3
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=csv_bytes))
        }
        
        loader = DataLoader()
        loader.aws_mode = True
        
        result_df = loader._read_csv_from_s3("8#Belt Conveyer")
        
        mock_s3.get_object.assert_called_once_with(
            Bucket='brilliant-automation-capstone',
            Key='processed/8#Belt Conveyer_merged.csv'
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 100
        assert 'Device' in result_df.columns
        assert 'datetime' in result_df.columns
    
    def test_local_loading_basic(self, tmp_path, sample_csv_data):
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        csv_file = processed_dir / "8#Belt Conveyer_merged.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        loader = DataLoader()
        loader.aws_mode = False
        loader._project_root = tmp_path
        
        loader.update_data("8#Belt Conveyer")
        
        result_df = loader.get_data("8#Belt Conveyer")
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 100
        assert 'timestamp' in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['timestamp'])
        assert 'Unnamed: 0' not in result_df.columns
    
    def test_device_name_normalization(self, tmp_path, sample_high_temp_fan_data):
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        csv_file = processed_dir / "1#High-Temp Fan_merged.csv"
        sample_high_temp_fan_data.to_csv(csv_file, index=False)
        
        loader = DataLoader()
        loader.aws_mode = False
        loader._project_root = tmp_path
        
        loader.update_data("1#High-Temp Fan")
        result_df = loader.get_data("1#High-Temp Fan")
        
        assert all(result_df['Device'] == "1#High-Temp Fan")
        assert "1#High-temperature Fan" not in result_df['Device'].values
    
    def test_time_filtering_basic(self, sample_csv_data):
        with patch('utils.data_loader.data_loader') as mock_data_loader:
            df = sample_csv_data.copy()
            df = df.rename(columns={'datetime': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            mock_data_loader.get_data.return_value = df
            
            start_time = "2023-01-01 10:30:00"
            end_time = "2023-01-01 12:00:00"
            
            result_df = load_data("8#Belt Conveyer", start_time, end_time)
            
            assert len(result_df) < len(df)
            assert result_df['timestamp'].min() >= pd.to_datetime(start_time)
            assert result_df['timestamp'].max() <= pd.to_datetime(end_time)
    
    def test_missing_file_error(self, tmp_path):
        loader = DataLoader()
        loader.aws_mode = False
        loader._project_root = tmp_path
        
        with pytest.raises(Exception):
            loader.update_data("NonExistent Device")
    
    @patch('boto3.client')
    def test_s3_error_handling(self, mock_boto3):
        mock_s3 = Mock()
        mock_boto3.return_value = mock_s3
        mock_s3.get_object.side_effect = Exception("S3 connection failed")
        
        loader = DataLoader()
        loader.aws_mode = True
        
        with pytest.raises(Exception, match="S3 connection failed"):
            loader._read_csv_from_s3("8#Belt Conveyer")
    
    def test_get_unique_functions(self, sample_csv_data):
        with patch('utils.data_loader.data_loader') as mock_data_loader:
            df = sample_csv_data.copy()
            mock_data_loader.get_data.return_value = df
            mock_data_loader.DEVICE_NAMES = ["8#Belt Conveyer", "1#High-Temp Fan", "Tube Mill"]
            
            locations = get_unique_locations("8#Belt Conveyer")
            expected_locations = sorted(df['location'].unique())
            assert locations == expected_locations
            
            devices = get_unique_devices()
            assert isinstance(devices, list)
            assert len(devices) == 3
    
    def test_data_transformations(self, tmp_path, sample_csv_data):
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        csv_file = processed_dir / "8#Belt Conveyer_merged.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        loader = DataLoader()
        loader.aws_mode = False
        loader._project_root = tmp_path
        
        loader.update_data("8#Belt Conveyer")
        result_df = loader.get_data("8#Belt Conveyer")
        
        assert 'timestamp' in result_df.columns
        assert 'datetime' not in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['timestamp'])
        assert 'Unnamed: 0' not in result_df.columns
        
        assert len(result_df) == len(sample_csv_data)
        assert all(col in result_df.columns for col in ['Device', 'location', 'Temperature'])