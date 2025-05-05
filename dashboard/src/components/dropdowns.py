from dash import dcc, html
import dash_bootstrap_components as dbc

# techdebt: using dummy options — replace with dynamic options from data later
DEVICE_OPTIONS = [
    {"label": f"Device {i}", "value": f"device_{i}"} for i in range(1, 4)
]
# todo: custom sets of sensor options based on which device is there
SENSOR_OPTIONS = [
    {"label": loc, "value": loc} for loc in ["Top", "Bottom", "Left", "Right"]
]

device_dropdown = html.Div([
    html.Label("Select Device"),
    dcc.Dropdown(
        id="device-dropdown",
        options=DEVICE_OPTIONS,
        value="device_1",
        clearable=False
    )
])

sensor_dropdown = html.Div([
    html.Label("Select Sensor Locations"),
    dcc.Dropdown(
        id="sensor-dropdown",
        options=SENSOR_OPTIONS,
        value=[opt["value"] for opt in SENSOR_OPTIONS],
        multi=True
    )
])
