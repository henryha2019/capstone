from dash import html, dcc
import dash_bootstrap_components as dbc
from components.dropdowns import device_dropdown, sensor_dropdown
from components.radar_chart import radar_chart_1, radar_chart_2
from components.overlay_plot import frequency_chart
from components.line_charts import detail_charts_column


def create_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Sensor Monitoring Dashboard"),
                html.Div("Last updated: 2025-04-30")
            ])
        ], className="my-3"),

        dbc.Row([
            dbc.Col(device_dropdown, width=4),
            dbc.Col(sensor_dropdown, width=8)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(radar_chart_1),
                    dbc.Col(radar_chart_2)
                ]),
                dbc.Row([
                    dbc.Col(frequency_chart)
                ])
            ], width=8),

            dbc.Col(detail_charts_column, width=4)
        ]),

        html.Hr(),

        dbc.Row([
            dbc.Col([
                html.Footer("Dashboard by Your Team — Data from XYZ Sensors Inc.")
            ])
        ], className="mt-4 text-center")
    ], fluid=True)
