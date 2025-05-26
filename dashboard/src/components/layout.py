from dash import html, dcc
import dash_bootstrap_components as dbc
from components.dropdowns import date_range_dropdown, device_dropdown, sensor_dropdown
from components.radar_chart import radar_chart_1, radar_chart_2
from components.overlay_plot import frequency_chart
from components.signal_charts import signal_charts_column
from components.header_timestamp import header_timestamp

def summary_view():
    return html.Div([
        # Radar charts
        dbc.Row([
            dbc.Col(radar_chart_1),
            dbc.Col(radar_chart_2)
        ], className="radar-row"),
        # Overlay plot
        dbc.Row([
            dbc.Col(frequency_chart)
        ], class_name="frequency-row")
    ], className="dashboard-col")

def signal_view():
    return html.Div([
        html.Div([
            dcc.Graph(id="signal-chart-sig-raw", className="signal-chart", config={"displayModeBar": False}),
            dcc.Graph(id="signal-chart-sig-fft", className="signal-chart", config={"displayModeBar": False}),
            dcc.Graph(id="signal-chart-env", className="signal-chart", config={"displayModeBar": False}),
            dcc.Graph(id="signal-chart-env-fft", className="signal-chart", config={"displayModeBar": False}),
        ], className="signal-row")
    ], style={"height": "100%", "width": "100%"})

def create_layout():
    return dbc.Container([
        # Timestamp
        dbc.Row([
            dbc.Col([
                header_timestamp()
            ]),
        ]),

        # Dropdowns
        dbc.Row([
            dbc.Col(date_range_dropdown, width=3),
            dbc.Col(device_dropdown, width=3),
            dbc.Col(sensor_dropdown, width=6)
        ]),

        dbc.Row([
            dbc.Col([
                dcc.Tabs(
                    id="view-tabs",
                    value="summary",
                    children=[
                        dcc.Tab(label="Summary View", value="summary"),
                        dcc.Tab(label="Signal View", value="signal"),
                    ]
                )
            ])
        ]),

        dbc.Row([
            dbc.Col([
                html.Div(id="summary-view", children=summary_view(), className="view-container"),
                html.Div(id="signal-view", children=signal_view(), className="view-container", style={"display": "none"})
            ], className="dashboard-col")
        ], className="dashboard-row")
    ], className="dash-container", fluid=True)