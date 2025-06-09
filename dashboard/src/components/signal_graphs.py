from dash import dcc
from components.signal_visualizer import SignalVisualizer
import plotly.graph_objects as go
from utils.config import FEATURES

STANDARD_NUMBER_OF_PLOTS = 4

def create_signal_graph(graph_id):
    return dcc.Graph(
        id=graph_id,
        figure=go.Figure(),
        className="signal-graph", 
        config={"displayModeBar": False}
    )

signal_graph_sig_raw = create_signal_graph("signal-graph-sig-raw")
signal_graph_sig_fft = create_signal_graph("signal-graph-sig-fft")
signal_graph_env = create_signal_graph("signal-graph-env")
signal_graph_env_fft = create_signal_graph("signal-graph-env-fft")

def update_signal_graphs(df):
    if df.empty or FEATURES['vibration_velocity_z'] not in df.columns:
        return [go.Figure()] * STANDARD_NUMBER_OF_PLOTS
    return SignalVisualizer(df).generate()