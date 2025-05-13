from dash import dcc
import plotly.express as px
import pandas as pd
from utils.plot_config import format_plot

frequency_chart = dcc.Graph(
    id="frequency-chart", 
    figure=px.line(), 
    className="graph-container",
    config={"displayModeBar": False}
    )

def update_frequency_chart(df):
    # techdebt: Placeholder logic
    fig = px.line(
        x=pd.date_range(start="2024-01-01", periods=50, freq="T"),
        y=[i % 10 for i in range(50)],
        title="Frequency Over Time"
    )
    return format_plot(fig)
