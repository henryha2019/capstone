from dash import html, dcc
import plotly.express as px
from utils.plot_config import format_plot

chart_ids = ["detail-chart-1", "detail-chart-2", "detail-chart-3", "detail-chart-4"]
detail_charts_column = html.Div([
    dcc.Graph(id=cid, className="detail-chart", config={"displayModeBar": False})
    for cid in chart_ids
], className="detail-charts-col")


def update_detail_charts(df):
    # Placeholder logic for 4 charts
    figures = []
    for i in range(4):
        fig = px.line(
            x=list(range(10)),
            y=[j * (i + 1) for j in range(10)],
            title=f"Detail Metric {i+1}"
        )
        fig = format_plot(fig)
        figures.append(fig)
    return figures
