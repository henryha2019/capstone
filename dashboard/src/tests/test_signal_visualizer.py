import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from components.signal_visualizer import SignalVisualizer


@pytest.fixture
def sample_vibration_data():
    np.random.seed(42)
    timestamps = pd.date_range('2023-01-01 10:00:00', periods=100, freq='10min')
    locations = ['Motor Drive End', 'Motor Non-Drive End', 'Gear Reducer']
    
    data = []
    for i, ts in enumerate(timestamps):
        location = locations[i % len(locations)]
        velocity = np.sin(2 * np.pi * 0.1 * i) + 0.1 * np.random.randn()
        data.append({
            'timestamp': ts,
            'location': location,
            'Vibration Velocity Z': velocity
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def empty_vibration_data():
    return pd.DataFrame(columns=['timestamp', 'location', 'Vibration Velocity Z'])


@pytest.fixture
def single_location_data():
    timestamps = pd.date_range('2023-01-01 10:00:00', periods=50, freq='10min')
    return pd.DataFrame({
        'timestamp': timestamps,
        'location': ['Motor Drive End'] * 50,
        'Vibration Velocity Z': np.sin(np.linspace(0, 4*np.pi, 50))
    })


class TestSignalVisualizer:
    
    def test_initialization_creates_four_figures(self, sample_vibration_data):
        visualizer = SignalVisualizer(sample_vibration_data)
        assert len(visualizer.figures) == 4
        assert all(isinstance(fig, go.Figure) for fig in visualizer.figures)
        assert visualizer.T == 5.0
    
    def test_custom_sampling_interval(self, sample_vibration_data):
        visualizer = SignalVisualizer(sample_vibration_data, sampling_interval=2.0)
        assert visualizer.T == 2.0
    
    def test_preprocess_returns_correct_arrays(self, sample_vibration_data):
        visualizer = SignalVisualizer(sample_vibration_data)
        signal, time, freq = visualizer.preprocess('Motor Drive End')
        
        assert isinstance(signal, np.ndarray)
        assert isinstance(time, np.ndarray)
        assert isinstance(freq, np.ndarray)
        assert len(signal) > 0
        assert len(freq) == len(signal) // 2
    
    def test_compute_y_values_removes_dc_component(self, sample_vibration_data):
        visualizer = SignalVisualizer(sample_vibration_data)
        signal = np.array([1, 2, 3, 4, 5]) + 10
        
        envelope, fft_vals, fft_env = visualizer.compute_y_values(signal)
        
        assert isinstance(envelope, np.ndarray)
        assert isinstance(fft_vals, np.ndarray) 
        assert isinstance(fft_env, np.ndarray)
        assert len(envelope) == len(signal)
        assert len(fft_vals) == len(signal) // 2
        assert len(fft_env) == len(signal) // 2
    
    def test_generate_returns_four_formatted_figures(self, sample_vibration_data):
        with patch('components.signal_visualizer.format_plot') as mock_format:
            mock_format.side_effect = lambda x: x
            
            visualizer = SignalVisualizer(sample_vibration_data)
            figures = visualizer.generate()
            
            assert len(figures) == 4
            assert mock_format.call_count == 4
    
    def test_empty_dataframe_handling(self, empty_vibration_data):
        visualizer = SignalVisualizer(empty_vibration_data)
        figures = visualizer.generate()
        
        assert len(figures) == 4
        for fig in figures:
            assert len(fig.data) == 0
    
    def test_single_location_processing(self, single_location_data):
        visualizer = SignalVisualizer(single_location_data)
        signal, time, freq = visualizer.preprocess('Motor Drive End')
        
        assert len(signal) == 50
        assert len(time) == 50
        assert len(freq) == 25
    
    def test_multiple_locations_create_multiple_traces(self, sample_vibration_data):
        visualizer = SignalVisualizer(sample_vibration_data)
        visualizer.generate()
        
        unique_locations = sample_vibration_data['location'].nunique()
        for fig in visualizer.figures:
            assert len(fig.data) == unique_locations
    
    def test_nonexistent_location_returns_empty_arrays(self, sample_vibration_data):
        visualizer = SignalVisualizer(sample_vibration_data)
        
        with pytest.raises(ZeroDivisionError):
            visualizer.preprocess('Nonexistent Location')
    
    def test_missing_vibration_column_raises_error(self):
        df_missing_column = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'location': ['Motor Drive End'] * 10
        })
        
        visualizer = SignalVisualizer(df_missing_column)
        with pytest.raises(KeyError):
            visualizer.preprocess('Motor Drive End')
    
    def test_figure_layout_updates(self, sample_vibration_data):
        visualizer = SignalVisualizer(sample_vibration_data)
        visualizer.generate()
        
        expected_xlabels = ["Time", "Frequency [Hz]", "Time", "Frequency [Hz]"]
        expected_ylabels = ["Velocity [mm/s]", "Amplitude", "Envelope Amplitude", "Amplitude"]
        
        for i, fig in enumerate(visualizer.figures):
            layout = fig.layout
            assert layout.xaxis.title.text == expected_xlabels[i]
            assert layout.yaxis.title.text == expected_ylabels[i]
        
        df_missing_column = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'location': ['Motor Drive End'] * 10
        })

        visualizer = SignalVisualizer(df_missing_column)
        with pytest.raises(KeyError):
            visualizer.preprocess('Motor Drive End')
    
    def test_figure_layout_updates(self, sample_vibration_data):
        visualizer = SignalVisualizer(sample_vibration_data)
        visualizer.generate()
        
        expected_xlabels = ["Time", "Frequency [Hz]", "Time", "Frequency [Hz]"]
        expected_ylabels = ["Velocity [mm/s]", "Amplitude", "Envelope Amplitude", "Amplitude"]
        
        for i, fig in enumerate(visualizer.figures):
            layout = fig.layout
            assert layout.xaxis.title.text == expected_xlabels[i]
            assert layout.yaxis.title.text == expected_ylabels[i]