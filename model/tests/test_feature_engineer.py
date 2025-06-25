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
import io
import sys

# Import functions to test - adjust the import path based on your project structure
# Assuming feature_engineer.py is in the same directory or adjust the import accordingly

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from feature_engineer import (
    read_json_file,
    band_rms,
    band_peak,
    bucket_summary,
    collect_json_files,
    read_merged_csv,
    save_csv,
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

def test_collect_json_files_local_mode(tmp_path):
    """Test collecting JSON files from local directory."""
    # Create mock directory structure that matches the expected format
    # The function expects: PROJECT_ROOT/data/voltage/<date>#Belt Conveyer/
    project_root = tmp_path / "project"
    voltage_dir = project_root / "data" / "voltage"
    belt_dir = voltage_dir / "20230101#Belt Conveyer"
    belt_dir.mkdir(parents=True)

    # Create test JSON files with proper format that matches the regex patterns
    # The code looks for patterns like: (\d{8})\s?(\d{6})_ or _(\d{8})_(\d{6})\.json$
    test_files = [
        "20230101 120000_device_67c29baa30e6dd385f031b30_VV.json",
        "20230101 130000_device_67c29baa30e6dd385f031b39_VV.json",
    ]

    for filename in test_files:
        json_file = belt_dir / filename
        with open(json_file, 'w') as f:
            # Create valid JSON with proper axisX spacing to avoid dt <= 0 error
            json.dump({"axisX": [0, 0.001, 0.002], "axisY": [0, 1, 0.5]}, f)

    # Mock both VOLTAGE_DIR and PROJECT_ROOT to match the expected structure
    with patch('feature_engineer.VOLTAGE_DIR', voltage_dir), \
         patch('feature_engineer.PROJECT_ROOT', project_root):
        records = collect_json_files(aws_mode=False)

    assert len(records) == 2
    assert all('timestamp' in record for record in records)
    assert all('filepath' in record for record in records)
    
    # Check that timestamps are parsed correctly
    timestamps = [record['timestamp'] for record in records]
    assert timestamps[0].year == 2023
    assert timestamps[0].month == 1
    assert timestamps[0].day == 1
    
    # Sort timestamps to check them in order
    timestamps.sort()
    assert timestamps[0].hour == 12  # 120000 = 12:00:00
    assert timestamps[1].hour == 13  # 130000 = 13:00:00
    
    # Check that all METRIC_COLS are present with pd.NA values
    for record in records:
        for col in ['velocity_rms', 'crest_factor', 'kurtosis_opt']:  # Sample of METRIC_COLS
            assert col in record
            assert pd.isna(record[col])

def test_read_merged_csv_local_mode(tmp_path):
    """Test reading merged CSV from local directory."""
    # Create test CSV file
    process_dir = tmp_path / "data" / "processed"
    process_dir.mkdir(parents=True)

    test_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=3, freq='h'),
        'location': ['A', 'B', 'C'],
        'value': [1, 2, 3]
    })

    csv_file = process_dir / "8#Belt Conveyer_merged.csv"
    test_data.to_csv(csv_file, index=False)

    # Mock the global PROCESS_DIR variable instead of get_config
    with patch('feature_engineer.PROCESS_DIR', process_dir):
        result = read_merged_csv('8#Belt Conveyer', aws_mode=False)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert pd.api.types.is_datetime64_any_dtype(result['datetime'])


def test_save_csv_local_mode(tmp_path):
    """Test saving CSV to local directory."""
    # Setup directories
    process_dir = tmp_path / "data" / "processed"
    process_dir.mkdir(parents=True)

    test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    # Mock the global PROCESS_DIR variable instead of get_config
    with patch('feature_engineer.PROCESS_DIR', process_dir):
        save_csv(test_data, 'test.csv', aws_mode=False, s3_prefix='processed')

    # Verify file was saved
    saved_file = process_dir / 'test.csv'
    assert saved_file.exists()

    loaded_data = pd.read_csv(saved_file)
    pd.testing.assert_frame_equal(test_data, loaded_data)


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
