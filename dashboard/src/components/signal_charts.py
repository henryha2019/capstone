from dash import html, dcc
from components.signal_visualizer import SignalVisualizer
import plotly.graph_objects as go

STANDARD_NUMBER_OF_PLOTS = 4
chart_ids = ["signal-chart-sig-raw", "signal-chart-sig-fft", "signal-chart-env", "signal-chart-env-fft"]

signal_charts_column = html.Div([
    dcc.Graph(id=cid, className="signal-chart", config={"displayModeBar": False})
    for cid in chart_ids
], className="signal-charts-col")


def update_signal_charts(df):
    # techdebt: config file where colnames can be put, currently fragile
    if df.empty or "Vibration Velocity Z" not in df.columns:
        return [go.Figure()] * STANDARD_NUMBER_OF_PLOTS
    return SignalVisualizer(df).generate()
