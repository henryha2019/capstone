import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import io
import os
import sys
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import from the src directory
from preprocess import (
    log_dataframe_metadata,
    create_datetime_columns,
    clean_dataframes,
    pivot_dataframes,
    process_temperature_data,
    filter_datetime_range,
    merge_dataframes,
    calculate_overlap_range,
    read_files_from_local,
    process_device_data
)


@pytest.fixture
def sample_features_df():
    """Sample features dataframe for testing"""
    return pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'Time': ['10:00:00', '11:00:00', '10:00:00', '11:00:00'],
        'Measurement': ['Temperature', 'Humidity', 'Temperature', 'Humidity'],
        'data': [25.5, 60.0, 26.0, 58.0],
        'location': ['Room1', 'Room1', 'Room2', 'Room2'],
        'id': [1, 2, 3, 4]
    })


@pytest.fixture
def sample_rating_df():
    """Sample rating dataframe for testing"""
    return pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'Time': ['10:30:00', '11:30:00', '10:30:00'],
        'Device': ['Device1', 'Device1', 'Device1'],
        'Metric': ['Performance', 'Performance', 'Performance'],
        'Rating': [8.5, 7.0, 9.0]
    })


@pytest.fixture
def sample_pivot_features_df():
    """Sample pivoted features dataframe"""
    return pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-02 10:00:00']),
        'location': ['Room1', 'Room1', 'Room2'],
        'Temperature': [25.5, np.nan, 26.0],
        'Humidity': [60.0, 65.0, 58.0]
    })


@pytest.fixture
def sample_pivot_rating_df():
    """Sample pivoted rating dataframe"""
    return pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 10:30:00', '2024-01-01 11:30:00', '2024-01-02 10:30:00']),
        'Device': ['Device1', 'Device1', 'Device1'],
        'Performance': [8.5, 7.0, 9.0]
    })


def test_log_dataframe_metadata(sample_features_df, caplog):
    """Test logging of dataframe metadata"""
    with caplog.at_level('INFO'):
        log_dataframe_metadata(sample_features_df, "Test DataFrame")

    assert "DATAFRAME SUMMARY: Test DataFrame" in caplog.text
    assert "Shape: (4, 6)" in caplog.text
    assert "Memory Usage:" in caplog.text


def test_create_datetime_columns(sample_features_df, sample_rating_df):
    """Test creation of datetime columns"""
    features_df, rating_df = create_datetime_columns(sample_features_df.copy(), sample_rating_df.copy())

    assert 'datetime' in features_df.columns
    assert 'datetime' in rating_df.columns
    assert pd.api.types.is_datetime64_any_dtype(features_df['datetime'])
    assert pd.api.types.is_datetime64_any_dtype(rating_df['datetime'])

    # Check specific datetime conversion
    expected_datetime = pd.to_datetime('2024-01-01 10:00:00')
    assert features_df['datetime'].iloc[0] == expected_datetime


def test_clean_dataframes(sample_features_df, sample_rating_df):
    """Test cleaning of dataframes by dropping columns"""
    # Add datetime column first
    sample_features_df['datetime'] = pd.to_datetime('2024-01-01 10:00:00')
    sample_rating_df['datetime'] = pd.to_datetime('2024-01-01 10:00:00')

    features_df, rating_df = clean_dataframes(sample_features_df.copy(), sample_rating_df.copy())

    # Check that specified columns are dropped
    assert 'Date' not in features_df.columns
    assert 'Time' not in features_df.columns
    assert 'id' not in features_df.columns

    assert 'Date' not in rating_df.columns
    assert 'Time' not in rating_df.columns

    # Check that other columns remain
    assert 'Measurement' in features_df.columns
    assert 'Device' in rating_df.columns


def test_pivot_dataframes():
    """Test pivoting of dataframes"""
    # Create test data with datetime column
    features_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 10:00:00']),
        'location': ['Room1', 'Room1'],
        'Measurement': ['Temperature', 'Humidity'],
        'data': [25.5, 60.0]
    })

    rating_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 10:30:00']),
        'Device': ['Device1'],
        'Metric': ['Performance'],
        'Rating': [8.5]
    })

    pivot_features_df, pivot_rating_df = pivot_dataframes(features_df, rating_df)

    # Check pivot structure
    assert 'Temperature' in pivot_features_df.columns
    assert 'Humidity' in pivot_features_df.columns
    assert 'Performance' in pivot_rating_df.columns
    assert pivot_features_df.shape[0] == 1  # Should have one row per datetime-location combination


def test_process_temperature_data(sample_pivot_features_df):
    """Test temperature data forward filling"""
    df = sample_pivot_features_df.copy()
    result_df = process_temperature_data(df)

    # Check that NaN temperature values are forward filled within location groups
    room1_temps = result_df[result_df['location'] == 'Room1']['Temperature']
    assert not room1_temps.isna().any()


def test_filter_datetime_range():
    """Test filtering dataframe by datetime range"""
    df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-02 10:00:00', '2024-01-03 10:00:00']),
        'value': [1, 2, 3]
    })

    start = pd.to_datetime('2024-01-01 10:00:00')
    end = pd.to_datetime('2024-01-02 10:00:00')

    filtered_df = filter_datetime_range(df, start, end)

    assert len(filtered_df) == 2
    assert filtered_df['datetime'].min() >= start
    assert filtered_df['datetime'].max() <= end


def test_calculate_overlap_range(sample_pivot_features_df, sample_pivot_rating_df):
    """Test calculation of overlapping datetime range"""
    overlap_start, overlap_end = calculate_overlap_range(sample_pivot_features_df, sample_pivot_rating_df)

    assert isinstance(overlap_start, pd.Timestamp)
    assert isinstance(overlap_end, pd.Timestamp)
    assert overlap_start <= overlap_end


def test_merge_dataframes():
    """Test merging of feature and rating dataframes"""
    features_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']),
        'location': ['Room1', 'Room1'],
        'Temperature': [25.5, 26.0]
    }).sort_values('datetime')

    rating_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 10:30:00', '2024-01-01 11:30:00']),
        'Device': ['Device1', 'Device1'],
        'Performance': [8.5, 7.0]
    }).sort_values('datetime')

    merged_df = merge_dataframes(features_df, rating_df)

    assert 'Temperature' in merged_df.columns
    assert 'Performance' in merged_df.columns
    assert len(merged_df) == 2

def test_read_files_from_local_no_files_found():
    """Test error handling when no files are found"""
    with patch('preprocess.glob.glob', return_value=[]):
        with pytest.raises(FileNotFoundError, match="No files found for device"):
            read_files_from_local('NonExistentDevice', '/path/to/data')


def test_read_files_from_local_no_rating_files():
    """Test error handling when no rating files are found"""
    with patch('preprocess.glob.glob', return_value=['/path/to/data/(TestDevice)_Location1.xlsx']):
        with pytest.raises(FileNotFoundError, match="No Rating file found"):
            read_files_from_local('TestDevice', '/path/to/data')
