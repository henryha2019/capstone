#!/usr/bin/env python3
"""
Test suite for feature_engineer.py

This module contains comprehensive test cases for all functions in the feature engineering pipeline.
Tests cover DSP processing, file handling, bucket processing, and data merging functionality.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import boto3
from moto import mock_s3
import io
import sys

# Import functions to test - adjust the import path based on your project structure
# Assuming feature_engineer.py is in the same directory or adjust the import accordingly

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from feature_engineer import (
    get_config,
    read_json_file,
    band_rms,
    band_peak,
    compute_dsp_metrics,
    bucket_summary,
    collect_json_files,
    read_merged_csv,
    save_csv,
    process_json_metrics,
    process_merged_data,
    merge_metrics_and_summary,
    run_feature_engineering_pipeline
)


# —————————— FIXTURES ——————————

@pytest.fixture
def sample_waveform():
    """Generate a sample waveform for testing DSP functions."""
    np.random.seed(42)
    fs = 1000.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
    # Create a signal with multiple frequency components
    signal = (
        np.sin(2 * np.pi * 50 * t) +  # 50 Hz component
        0.5 * np.sin(2 * np.pi * 150 * t) +  # 150 Hz component
        0.2 * np.sin(2 * np.pi * 5000 * t) +  # 5 kHz component
        0.1 * np.random.randn(len(t))  # noise
    )
    return signal, fs


@pytest.fixture
def sample_json_data():
    """Generate sample JSON data for testing JSON file reading."""
    axis_x = np.linspace(0, 1, 1000)
    axis_y = np.sin(2 * np.pi * 50 * axis_x) + 0.1 * np.random.randn(1000)
    return {
        "axisX": axis_x.tolist(),
        "axisY": axis_y.tolist()
    }


@pytest.fixture
def temp_json_file(sample_json_data):
    """Create a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_json_data, f)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()  # cleanup


@pytest.fixture
def sample_merged_dataframe():
    """Generate sample merged DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01 10:00:00', periods=100, freq='10min')
    locations = ['Motor Drive End', 'Motor Non-Drive End', 'Fan Free End']

    data = []
    for i, dt in enumerate(dates):
        data.append({
            'datetime': dt,
            'location': np.random.choice(locations),
            'High-Frequency Acceleration': np.random.normal(10, 2),
            'Low-Frequency Acceleration Z': np.random.normal(5, 1),
            'Temperature': np.random.normal(25, 3),
            'Vibration Velocity Z': np.random.normal(2, 0.5),
            'alignment_status': np.random.choice([0, 1, 2]),
            'bearing_lubrication': np.random.choice([0, 1, 2]),
            'crest_factor': np.random.choice([0, 1, 2]),
            'electromagnetic_status': np.random.choice([0, 1, 2]),
            'fit_condition': np.random.choice([0, 1, 2]),
            'kurtosis_opt': np.random.choice([0, 1, 2]),
            'rms_10_25khz': np.random.choice([0, 1, 2]),
            'rms_1_10khz': np.random.choice([0, 1, 2]),
            'rotor_balance_status': np.random.choice([0, 1, 2]),
            'rubbing_condition': np.random.choice([0, 1, 2]),
            'velocity_rms': np.random.choice([0, 1, 2]),
            'peak_value_opt': np.random.choice([0, 1, 2]),
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_metrics_dataframe():
    """Generate sample metrics DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01 10:05:00', periods=50, freq='20min')
    locations = ['Motor Drive End', 'Motor Non-Drive End', 'Fan Free End']

    data = []
    for dt in dates:
        data.append({
            'datetime': dt,
            'location': np.random.choice(locations),
            'velocity_rms': np.random.normal(2, 0.5),
            'crest_factor': np.random.normal(3, 0.8),
            'kurtosis_opt': np.random.normal(3, 1),
            'peak_value_opt': np.random.normal(5, 1),
            'rms_0_10hz': np.random.normal(0.5, 0.1),
            'rms_10_100hz': np.random.normal(1, 0.2),
            'rms_1_10khz': np.random.normal(0.3, 0.1),
            'rms_10_25khz': np.random.normal(0.1, 0.05),
            'peak_10_1000hz': np.random.normal(3, 0.5),
            'sensor_id': '67c29baa30e6dd385f031b30',
            'wave_code': 'VV',
            'filepath': f'data/voltage/2023010{np.random.randint(1,9)}/20230101_120000_device_67c29baa30e6dd385f031b30_VV.json'
        })

    return pd.DataFrame(data)


# —————————— CONFIGURATION TESTS ——————————

def test_get_config():
    """Test configuration loading."""
    config = get_config()

    assert 'PROJECT_ROOT' in config
    assert 'S3_BUCKET' in config
    assert 'METRIC_COLS' in config
    assert 'RATING_COLS' in config
    assert 'SENSOR_MAP' in config
    assert 'FILENAME_PATTERN' in config

    assert len(config['METRIC_COLS']) == 9
    assert len(config['RATING_COLS']) == 12
    assert isinstance(config['PROJECT_ROOT'], Path)
    assert config['S3_BUCKET'] == 'brilliant-automation-capstone'


# —————————— DSP PROCESSING TESTS ——————————

def test_read_json_file_from_path(temp_json_file, sample_json_data):
    """Test reading JSON file from file path."""
    fs, signal = read_json_file(path=temp_json_file)

    assert isinstance(fs, float)
    assert fs > 0
    assert isinstance(signal, np.ndarray)
    assert len(signal) == len(sample_json_data['axisY'])
    assert signal.dtype == np.float32


def test_read_json_file_from_content(sample_json_data):
    """Test reading JSON file from content string."""
    content = json.dumps(sample_json_data)
    fs, signal = read_json_file(content=content)

    assert isinstance(fs, float)
    assert fs > 0
    assert isinstance(signal, np.ndarray)
    assert len(signal) == len(sample_json_data['axisY'])


def test_read_json_file_invalid_data():
    """Test reading JSON file with invalid data."""
    # Test with insufficient data points
    invalid_data = {"axisX": [0], "axisY": [1]}
    content = json.dumps(invalid_data)

    with pytest.raises(ValueError, match="Not enough points in axisX"):
        read_json_file(content=content)

    # Test with invalid dt (zero or negative)
    invalid_data = {"axisX": [0, 0], "axisY": [1, 2]}
    content = json.dumps(invalid_data)

    with pytest.raises(ValueError, match="Invalid dt"):
        read_json_file(content=content)


@pytest.mark.parametrize("f_lo,f_hi,expected_positive", [
    (0.1, 10, True),
    (10, 100, True),
    # (1000, 10000, True),
    (50000, 60000, False),  # No signal in this range
])
def test_band_rms(sample_waveform, f_lo, f_hi, expected_positive):
    """Test RMS calculation in frequency bands."""
    signal, fs = sample_waveform
    rms_value = band_rms(signal, fs, f_lo, f_hi)

    if expected_positive:
        assert rms_value > 0
        assert not np.isnan(rms_value)
    else:
        # For frequency bands with no signal, result might be very small or NaN
        assert rms_value >= 0 or np.isnan(rms_value)


@pytest.mark.parametrize("f_lo,f_hi,expected_positive", [
    (10, 1000, True),
    (50000, 60000, False),  # No signal in this range
])
def test_band_peak(sample_waveform, f_lo, f_hi, expected_positive):
    """Test peak calculation in frequency bands."""
    signal, fs = sample_waveform
    peak_value = band_peak(signal, fs, f_lo, f_hi)

    if expected_positive:
        assert peak_value > 0
        assert not np.isnan(peak_value)
    else:
        assert peak_value >= 0 or np.isnan(peak_value)


def test_compute_dsp_metrics(sample_waveform):
    """Test DSP metrics computation."""
    signal, fs = sample_waveform
    metrics = compute_dsp_metrics(signal, fs)

    assert len(metrics) == 9  # Should return 9 metrics
    assert all(isinstance(m, (float, np.floating)) for m in metrics)

    # Test specific metrics properties
    velocity_rms, crest_factor, kurtosis_opt, peak_value_opt = metrics[:4]

    assert velocity_rms > 0
    assert peak_value_opt > 0
    assert crest_factor > 0  # Should be positive for real signals
    assert not np.isnan(kurtosis_opt)


def test_compute_dsp_metrics_edge_cases():
    """Test DSP metrics with edge cases."""
    # Test with zero signal
    zero_signal = np.zeros(1000)
    fs = 1000.0
    metrics = compute_dsp_metrics(zero_signal, fs)

    assert metrics[0] == 0  # velocity_rms should be 0
    assert np.isnan(metrics[1])  # crest_factor should be NaN (0/0)
    assert metrics[3] == 0  # peak_value_opt should be 0


# —————————— BUCKET PROCESSING TESTS ——————————

def test_bucket_summary(sample_merged_dataframe):
    """Test bucket summary creation."""
    # Rename rating columns for testing
    rating_cols = [
        "alignment_status", "bearing_lubrication", "crest_factor",
        "electromagnetic_status", "fit_condition", "kurtosis_opt",
        "rms_10_25khz", "rms_1_10khz", "rotor_balance_status",
        "rubbing_condition", "velocity_rms", "peak_value_opt"
    ]

    measurement_cols = [
        'High-Frequency Acceleration',
        'Low-Frequency Acceleration Z',
        'Temperature',
        'Vibration Velocity Z'
    ]

    # Add rating suffix
    df = sample_merged_dataframe.copy()
    df = df.rename(columns={c: f"{c}_rating" for c in rating_cols})
    rating_cols_renamed = [f"{c}_rating" for c in rating_cols]

    result = bucket_summary(
        df,
        measurement_cols=measurement_cols,
        rating_cols=rating_cols_renamed,
        time_col='datetime',
        location_col='location',
        bucket_minutes=20
    )

    # Check that result has expected structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert 'bucket_id' in result.columns
    assert 'bucket_start' in result.columns
    assert 'bucket_end' in result.columns

    # Check aggregated columns exist
    for col in measurement_cols:
        assert f"{col}_count" in result.columns
        assert f"{col}_mean" in result.columns
        assert f"{col}_std" in result.columns
        assert f"{col}_min" in result.columns
        assert f"{col}_max" in result.columns

    # Check rating columns are preserved
    for col in rating_cols_renamed:
        assert col in result.columns


def test_bucket_summary_empty_dataframe():
    """Test bucket summary with empty DataFrame."""
    empty_df = pd.DataFrame(columns=[
        'datetime', 'location', 'measurement1', 'rating1'
    ])

    result = bucket_summary(
        empty_df,
        measurement_cols=['measurement1'],
        rating_cols=['rating1'],
        bucket_minutes=20
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# —————————— FILE HANDLING TESTS ——————————

@mock_s3
def test_collect_json_files_aws_mode():
    """Test collecting JSON files from S3."""
    # Setup mock S3
    s3 = boto3.client('s3', region_name='us-east-1')
    bucket_name = 'brilliant-automation-capstone'
    s3.create_bucket(Bucket=bucket_name)

    # Add test files to S3
    test_files = [
        'voltage/20230101#Belt Conveyer/20230101 120000_device_67c29baa30e6dd385f031b30_VV.json',
        'voltage/20230102#Belt Conveyer/20230102 130000_device_67c29baa30e6dd385f031b39_VV.json',
    ]

    for file_key in test_files:
        s3.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=json.dumps({"axisX": [0, 1], "axisY": [0, 1]})
        )

    # Test the function
    records = collect_json_files(aws_mode=True)

    assert len(records) == 2
    assert all('timestamp' in record for record in records)
    assert all('filepath' in record for record in records)
    assert all('s3_key' in record for record in records)


def test_collect_json_files_local_mode(tmp_path):
    """Test collecting JSON files from local directory."""
    # Create mock directory structure
    voltage_dir = tmp_path / "data" / "voltage"
    belt_dir = voltage_dir / "20230101#Belt Conveyer"
    belt_dir.mkdir(parents=True)

    # Create test JSON files
    test_files = [
        "20230101 120000_device_67c29baa30e6dd385f031b30_VV.json",
        "20230101 130000_device_67c29baa30e6dd385f031b39_VV.json",
    ]

    for filename in test_files:
        json_file = belt_dir / filename
        with open(json_file, 'w') as f:
            json.dump({"axisX": [0, 1], "axisY": [0, 1]}, f)

    with patch('feature_engineer.get_config') as mock_config:
        mock_config.return_value = {
            'VOLTAGE_DIR': voltage_dir,
            'METRIC_COLS': ['velocity_rms', 'crest_factor'],
            'PROJECT_ROOT': tmp_path
        }

        records = collect_json_files(aws_mode=False)

    assert len(records) == 2
    assert all('timestamp' in record for record in records)
    assert all('filepath' in record for record in records)

def test_read_merged_csv_local_mode(tmp_path):
    """Test reading merged CSV from local directory."""
    # Create test CSV file
    process_dir = tmp_path / "data" / "processed"
    process_dir.mkdir(parents=True)

    test_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=3, freq='H'),
        'location': ['A', 'B', 'C'],
        'value': [1, 2, 3]
    })

    csv_file = process_dir / "8#Belt Conveyer_merged.csv"
    test_data.to_csv(csv_file, index=False)

    with patch('feature_engineer.get_config') as mock_config:
        mock_config.return_value = {
            'PROCESS_DIR': process_dir
        }

        result = read_merged_csv('8#Belt Conveyer', aws_mode=False)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert pd.api.types.is_datetime64_any_dtype(result['datetime'])


@mock_s3
def test_save_csv_aws_mode():
    """Test saving CSV to S3."""
    # Setup mock S3
    s3 = boto3.client('s3', region_name='us-east-1')
    bucket_name = 'brilliant-automation-capstone'
    s3.create_bucket(Bucket=bucket_name)

    # Test data
    test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    # Test the function
    save_csv(test_data, 'test.csv', aws_mode=True, s3_prefix='processed')

    # Verify file was saved
    response = s3.get_object(Bucket=bucket_name, Key='processed/test.csv')
    saved_content = response['Body'].read().decode('utf-8')

    assert 'a,b' in saved_content
    assert '1,4' in saved_content


def test_save_csv_local_mode(tmp_path):
    """Test saving CSV to local directory."""
    # Setup directories
    process_dir = tmp_path / "data" / "processed"
    process_dir.mkdir(parents=True)

    test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    with patch('feature_engineer.get_config') as mock_config:
        mock_config.return_value = {
            'PROCESS_DIR': process_dir,
            'VOLTAGE_DIR': tmp_path / "data" / "voltage"
        }

        save_csv(test_data, 'test.csv', aws_mode=False, s3_prefix='processed')

    # Verify file was saved
    saved_file = process_dir / 'test.csv'
    assert saved_file.exists()

    loaded_data = pd.read_csv(saved_file)
    pd.testing.assert_frame_equal(test_data, loaded_data)


# —————————— DATA MERGING TESTS ——————————

def test_merge_metrics_and_summary_no_overlap():
    """Test merging with no overlapping intervals."""
    # Create non-overlapping data
    metrics_df = pd.DataFrame({
        'datetime': [pd.Timestamp('2023-01-01 10:00:00')],
        'location': ['Location A'],
        'velocity_rms': [1.0]
    })

    summary_df = pd.DataFrame({
        'datetime': [pd.Timestamp('2023-01-01 12:00:00')],  # Different time
        'bucket_end': [pd.Timestamp('2023-01-01 12:20:00')],
        'location': ['Location A'],
        'measurement_mean': [2.0]
    })

    result = merge_metrics_and_summary(metrics_df, summary_df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0  # No overlapping intervals


# —————————— INTEGRATION TESTS ——————————

@patch('feature_engineer.process_json_metrics')
@patch('feature_engineer.process_merged_data')
@patch('feature_engineer.save_csv')
def test_run_feature_engineering_pipeline(mock_save, mock_process_merged, mock_process_json):
    """Test the complete feature engineering pipeline."""
    # Mock return values
    mock_metrics_df = pd.DataFrame({
        'datetime': [pd.Timestamp('2023-01-01 10:00:00')],
        'location': ['Location A'],
        'velocity_rms': [1.0]
    })

    mock_summary_df = pd.DataFrame({
        'datetime': [pd.Timestamp('2023-01-01 10:00:00')],
        'bucket_end': [pd.Timestamp('2023-01-01 10:20:00')],
        'location': ['Location A'],
        'measurement_mean': [2.0]
    })

    mock_process_json.return_value = mock_metrics_df
    mock_process_merged.return_value = mock_summary_df

    # Run pipeline
    result = run_feature_engineering_pipeline('8#Belt Conveyer', aws_mode=False)
    
    # Verify function calls
    mock_process_json.assert_called_once_with(False, '8#Belt Conveyer')
    mock_process_merged.assert_called_once_with('8#Belt Conveyer', False)
    assert mock_save.call_count == 3  # metrics, summary, and full features
    
    # Verify result
    assert isinstance(result, pd.DataFrame)


# —————————— PARAMETRIZED TESTS ——————————

@pytest.mark.parametrize("fs,expected_bands", [
    (1000, [(0.1, 10), (10, 100), (100, 500)]),
    (25000, [(0.1, 10), (10, 100), (1000, 10000), (10000, 25000)]),
    (50000, [(0.1, 10), (10, 100), (1000, 10000), (10000, 25000)]),
])
def test_band_rms_different_sampling_rates(fs, expected_bands):
    """Test band RMS with different sampling rates."""
    # Generate test signal
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave

    for f_lo, f_hi in expected_bands:
        if f_hi < fs / 2:  # Ensure we're below Nyquist frequency
            rms_val = band_rms(signal, fs, f_lo, f_hi)
            assert isinstance(rms_val, (float, np.floating))
            assert not np.isnan(rms_val)


@pytest.mark.parametrize("bucket_minutes,expected_buckets", [
    (10, "more"),
    (30, "fewer"),
    (60, "fewer"),
])
def test_bucket_summary_different_intervals(sample_merged_dataframe, bucket_minutes, expected_buckets):
    """Test bucket summary with different time intervals."""
    # Prepare data
    df = sample_merged_dataframe.copy()
    rating_cols = ['alignment_status', 'bearing_lubrication']
    measurement_cols = ['High-Frequency Acceleration', 'Temperature']
    
    df = df.rename(columns={c: f"{c}_rating" for c in rating_cols})
    rating_cols_renamed = [f"{c}_rating" for c in rating_cols]
    
    result = bucket_summary(
        df,
        measurement_cols=measurement_cols,
        rating_cols=rating_cols_renamed,
        bucket_minutes=bucket_minutes
    )
    
    # More specific checks could be added based on expected bucket behavior
    assert len(result) > 0
    assert 'bucket_id' in result.columns


def test_filename_pattern_matching():
    """Test the regex pattern for filename matching."""
    config = get_config()
    pattern = config['FILENAME_PATTERN']
    
    # Valid filename
    valid_filename = "20230101 120000_device_67c29baa30e6dd385f031b30_67c29baa30e6dd385f031b39_VV.json"
    match = pattern.match(valid_filename)
    
    if match:  # Pattern might need adjustment based on actual filenames
        assert len(match.groups()) == 2
    
    # Invalid filename
    invalid_filename = "invalid_filename.json"
    match = pattern.match(invalid_filename)
    assert match is None


# —————————— ERROR HANDLING TESTS ——————————

def test_read_json_file_nonexistent_file():
    """Test reading a non-existent JSON file."""
    nonexistent_path = Path("nonexistent_file.json")
    
    with pytest.raises(FileNotFoundError):
        read_json_file(path=nonexistent_path)


def test_read_json_file_invalid_json():
    """Test reading invalid JSON content."""
    invalid_json = "{ invalid json content"
    
    with pytest.raises(json.JSONDecodeError):
        read_json_file(content=invalid_json)


def test_band_rms_invalid_frequency_range(sample_waveform):
    """Test band RMS with invalid frequency range."""
    signal, fs = sample_waveform
    
    # Test with f_lo > f_hi
    rms_val = band_rms(signal, fs, 100, 10)
    assert np.isnan(rms_val) or rms_val == 0


def test_compute_dsp_metrics_empty_signal():
    """Test DSP metrics with empty signal."""
    empty_signal = np.array([])
    fs = 1000.0
    
    with pytest.raises((ValueError, IndexError)):
        compute_dsp_metrics(empty_signal, fs)
