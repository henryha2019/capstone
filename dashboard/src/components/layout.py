from dash import html
import dash_bootstrap_components as dbc
from components.dropdowns import device_dropdown, sensor_dropdown
from components.radar_chart import radar_chart_1, radar_chart_2
from components.overlay_plot import frequency_chart
from components.signal_charts import signal_charts_column
from components.header_timestamp import header_timestamp

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
            dbc.Col(device_dropdown, width=4),
            dbc.Col(sensor_dropdown, width=8)
        ]),

        dbc.Row([
            dbc.Col([
                html.Div([
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
            ], width=8),

            # Signal charts
            dbc.Col(signal_charts_column, width=4, className="dashboard-col")
        ], className="dashboard-row"),

    ], className="dash-container", fluid=True)
