from dash import dcc
import plotly.graph_objs as go
from utils.plot_config import format_plot
from utils.config import LOCATION_COLOUR_MAP, RATING_COLOUR_MAP

rating_health_graph = dcc.Graph(
    id="rating-health-graph",
    figure=go.Figure(),
    className="graph-container",
    config={"displayModeBar": False}
)

high_frequency_graph = dcc.Graph(
    id="frequency-graph",
    figure=go.Figure(),
    className="graph-container",
    config={"displayModeBar": False}
)

def create_overlay_figure(df, y_columns, y_label):
    traces = []
    
    is_rating_graph = "Rating Health" in y_label
    
    for col in y_columns:
        if col not in df.columns:
            continue  # Skip columns that are not present
        for loc in df["location"].unique():
            subset = df[df["location"] == loc]
            
            if is_rating_graph:
                color = RATING_COLOUR_MAP.get(col, "#1f77b4")
                name = f"{loc} - {col.replace('_', ' ').title()}"
            else:
                color = LOCATION_COLOUR_MAP.get(loc, "#1f77b4")
                name = f"{loc} - {col}"
            
            traces.append(go.Scatter(
                x=subset["timestamp"],
                y=subset[col],
                mode="lines",
                name=name,
                line=dict(color=color)
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=y_label,
        margin=dict(l=40, r=20, t=10, b=40),
        hovermode="closest"
    )
    return format_plot(fig)