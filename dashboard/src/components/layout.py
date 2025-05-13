from dash import html
import dash_bootstrap_components as dbc
from components.dropdowns import device_dropdown, sensor_dropdown
from components.radar_chart import radar_chart_1, radar_chart_2
from components.overlay_plot import frequency_chart
from components.line_charts import detail_charts_column


def create_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("2025-04-30 15:23:58", className="dashboard-header")
            ]),
        ]),

        dbc.Row([
            dbc.Col(device_dropdown, width=4),
            dbc.Col(sensor_dropdown, width=8)
        ]),

        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        dbc.Col(radar_chart_1),
                        dbc.Col(radar_chart_2)
                    ], className="radar-row"),
                    dbc.Row([
                        dbc.Col(frequency_chart)
                    ], class_name="frequency-row")
                ], className="dashboard-col")
            ], width=8),

            dbc.Col(detail_charts_column, width=4, className="dashboard-col")
        ], className="dashboard-row"),

    ], className="dash-container", fluid=True)
