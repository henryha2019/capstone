import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
import plotly.graph_objects as go
from utils.plot_config import format_plot
from utils.config import LOCATION_COLOUR_MAP, FEATURES

class SignalVisualizer:
    """
    Generates time-domain and frequency-domain visualizations of vibration signals
    from multiple sensor locations.

    Take raw vibration velocity data using FFT and Hilbert transforms
    to produce four standard plots per sensor:
      1. Raw vibration velocity over time
      2. Frequency spectrum (FFT) of the raw signal
      3. Envelope signal over time (via Hilbert transform)
      4. Frequency spectrum (FFT) of the envelope signal

    Attributes:
        df (pd.DataFrame): Input DataFrame containing 'timestamp', 'location', and 'Vibration Velocity Z' columns.
        T (float): Sampling interval in seconds.
        figures (list): List of Plotly `go.Figure` objects, one per graph.

    Methods:
        preprocess(location):
            Extracts and sorts signal and time arrays for a given sensor location.

        compute_y_values(signal):
            Computes envelope, FFT of signal, and FFT of the envelope.

        add_traces(location):
            Adds all four visualizations for the specified sensor location to the figure list.

        generate():
            Iterates over all locations in the data and returns formatted Plotly figures for display.
    """

    def __init__(self, df, sampling_interval=5.0):
        self.df = df
        self.T = sampling_interval
        self.figures = [go.Figure() for _ in range(4)]

    def preprocess(self, location):
        df_loc = self.df[self.df["location"] == location].sort_values("timestamp")
        signal = df_loc[FEATURES["vibration_velocity_z"]].dropna().values
        time = df_loc["timestamp"].values
        N = len(signal)
        freq = fftfreq(N, self.T)[:N//2]
        return signal, time, freq

    def compute_y_values(self, signal):
        # Remove DC component from signal by subtracting mean
        signal = signal - np.mean(signal)
        envelope = np.abs(hilbert(signal))
        # Do the same with the envelope
        envelope = envelope - np.mean(envelope)
        fft_vals = np.abs(fft(signal))[:len(signal)//2]
        fft_env = np.abs(fft(envelope))[:len(signal)//2]
        return envelope, fft_vals, fft_env

    def add_traces(self, location):
        signal, time, freq = self.preprocess(location)
        envelope, fft_vals, fft_env = self.compute_y_values(signal)

        graph_data = {
            0: (time, signal, "Time", "Velocity [mm/s]"),
            1: (freq, fft_vals, "Frequency [Hz]", "Amplitude"),
            2: (time, envelope, "Time", "Envelope Amplitude"),
            3: (freq, fft_env, "Frequency [Hz]", "Amplitude"),
        }

        for i, (x, y, xlabel, ylabel) in graph_data.items():
            self.figures[i].add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=location,
                line=dict(color=LOCATION_COLOUR_MAP.get(location))
            ))
            self.figures[i].update_layout(xaxis_title=xlabel, yaxis_title=ylabel)

    def generate(self):
        for location in self.df["location"].unique():
            self.add_traces(location)
        return [format_plot(fig) for fig in self.figures]
