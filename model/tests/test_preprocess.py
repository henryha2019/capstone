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
    filter_datetime_range,
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


def test_log_dataframe_metadata(sample_features_df, caplog):
    """Test logging of dataframe metadata"""
    with caplog.at_level('INFO'):
        log_dataframe_metadata(sample_features_df, "Test DataFrame")

    assert "DATAFRAME SUMMARY: Test DataFrame" in caplog.text
    assert "Shape: (4, 6)" in caplog.text
    assert "Memory Usage:" in caplog.text


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
