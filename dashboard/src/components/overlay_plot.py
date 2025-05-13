from dash import dcc
import plotly.graph_objs as go
from utils.plot_config import format_plot
from utils.colours import COLOUR_MAP

frequency_chart = dcc.Graph(
    id="frequency-chart", 
    figure=go.Figure(), 
    className="graph-container",
    config={"displayModeBar": False}
    )

def update_frequency_chart(df):
    traces = []

    for loc in df["location"].unique():
        subset = df[df["location"] == loc]
        traces.append(go.Scatter(
            x=subset["timestamp"],
            # todo: check if we want to have this overlay for more than just HFA
            y=subset["High-Frequency Acceleration"],
            mode="lines",
            name=loc,
            line=dict(color=COLOUR_MAP.get(loc))
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="High-Frequency Acceleration (a.u.)",
        margin=dict(l=40, r=20, t=10, b=40),
        hovermode="closest"
    )
    return format_plot(fig)
