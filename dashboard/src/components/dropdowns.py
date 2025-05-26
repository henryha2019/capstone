from dash import dcc, html
from utils.data_loader import get_unique_devices, get_unique_locations
from utils.colours import COLOUR_EMOJI

def create_date_range_dropdown():
    return dcc.Dropdown(
        id='date-range-dropdown',
        options=[
            {'label': 'Last 20 minutes', 'value': '20min'},
            {'label': '24 hours', 'value': '24h'},
            {'label': 'Last 7 days', 'value': '7d'},
        ],
        value='20min',
        clearable=False
    )

def create_device_dropdown():
    devices = get_unique_devices()
    return dcc.Dropdown(
        id="device-dropdown",
        options=[{"label": device, "value": device} for device in devices],
        value=devices[0] if devices else None,
        clearable=False
    )

def create_sensor_dropdown():
    locations = get_unique_locations()
    return dcc.Dropdown(
        id="sensor-dropdown",
        options=[{"label": f"{COLOUR_EMOJI.get(loc, '')} {loc}", "value": loc} for loc in locations],
        value=locations,
        multi=True
    )

date_range_dropdown = create_date_range_dropdown()
device_dropdown = create_device_dropdown()
sensor_dropdown = create_sensor_dropdown()
