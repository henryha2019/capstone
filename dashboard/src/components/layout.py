from dash import html, dcc
import dash_bootstrap_components as dbc
from .dropdowns import create_date_range_dropdown, device_dropdown, dual_dropdown
from .radar_graphs import radar_graph_1, radar_graph_2
from .overlay_plot import rating_health_graph, high_frequency_graph
from .signal_graphs import signal_graph_sig_raw, signal_graph_env, signal_graph_sig_fft, signal_graph_env_fft
from .header_timestamp import header_timestamp

def ratings_view():
    return[
        # Rating radar graphs
        dbc.Row([
            dbc.Col(radar_graph_1),
            dbc.Col(radar_graph_2)
        ], className="graph-row"),

        # Rating overlay
        dbc.Row([
            dbc.Col(rating_health_graph)
        ], className="graph-row")
    ]

def sensor_view():
    return [
        # High frequency overlay
        dbc.Row([
            dbc.Col(high_frequency_graph)
        ], className="graph-row"),

        # Signal graphs
        dbc.Row([
            dbc.Col(signal_graph_sig_raw),
            dbc.Col(signal_graph_sig_fft),
        ], className="graph-row"),
        dbc.Row([
            dbc.Col(signal_graph_env),
            dbc.Col(signal_graph_env_fft),
        ], className="graph-row")
    ]


def create_layout():
    return dbc.Container([
        # Timestamp
        dbc.Row([
            dbc.Col(html.Img(src="assets/logo.png", height="50px"), width=4),
            dbc.Col([header_timestamp()]),
        ]),

        # Device and Time Dropdowns
        dbc.Row([
            dbc.Col(device_dropdown, width=4, className="dropdown-input"),
            dbc.Col(create_date_range_dropdown("start"), width=4),
            dbc.Col(create_date_range_dropdown("end"), width=4),
        ]),

        # Tabs
        dbc.Row([
            dbc.Col([
                dcc.Tabs(
                    id="view-tabs",
                    value="ratings",
                    children=[
                        dcc.Tab(label="Ratings Data", value="ratings"),
                        dcc.Tab(label="Sensor Location Data", value="sensor"),
                    ]
                )
            ])
        ]),

        # Rating/Location Dropdowns
        dbc.Row([
            dbc.Col(dual_dropdown, width=12, className="dropdown-input"),
        ]),

        # Graphs
        dbc.Row([
            dbc.Col([
                html.Div(id="ratings-view", children=ratings_view(), className="tab-content-container"),
                html.Div(id="signal-view", children=sensor_view(), className="tab-content-container", style={"display": "none"})
            ])
        ])
    ], className="dash-container", fluid=True)