from dash import dcc, html
import dash_bootstrap_components as dbc
from utils.data_loader import get_unique_devices, get_unique_locations, data_loader
from utils.colours import COLOUR_EMOJI

def create_date_range_dropdown(id_prefix):
    df = data_loader.get_data()
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()

    date = min_ts.date() if id_prefix == "start" else max_ts.date()
    time = min_ts.strftime("%H:%M") if id_prefix == "start" else max_ts.strftime("%H:%M")

    return dbc.InputGroup([
        dcc.DatePickerSingle(
            id=f"{id_prefix}-date",
            min_date_allowed=min_ts.date(),
            max_date_allowed=max_ts.date(),
            date=date
        ),
        dbc.Input(
            id=f"{id_prefix}-time",
            type="time",
            value=time,
            debounce=True
        )
    ], className="datetime-selector")


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

device_dropdown = create_device_dropdown()
sensor_dropdown = create_sensor_dropdown()
