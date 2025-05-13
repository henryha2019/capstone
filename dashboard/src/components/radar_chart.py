from dash import dcc
import plotly.express as px
import pandas as pd
from utils.plot_config import format_plot

# techdebt: Placeholder chart instance
radar_chart_1 = dcc.Graph(
    id="radar-chart-1", 
    figure=px.line_polar(), 
    className="graph-container", 
    config={"displayModeBar": False}
    )
radar_chart_2 = dcc.Graph(
    id="radar-chart-2", 
    figure=px.line_polar(), 
    className="graph-container", 
    config={"displayModeBar": False}
    )


def update_radar_chart(df, chart_id):
    # techdebt: Placeholder logic
    fig = px.line_polar(
        r=[1, 2, 3, 4],
        theta=["High Freq", "Low Freq", "Vibration", "Temperature"],
        line_close=True,
        title=f"Radar Chart {chart_id}"
    )
    return format_plot(fig)
