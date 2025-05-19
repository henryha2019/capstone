from dash import dcc, html
from utils.data_loader import get_unique_devices, get_unique_locations
from utils.colours import COLOUR_EMOJI

def create_device_dropdown():
    """Create the device dropdown component"""
    devices = get_unique_devices()
    return dcc.Dropdown(
        id="device-dropdown",
        options=[{"label": device, "value": device} for device in devices],
        value=devices[0] if devices else None,
        clearable=False
    )

def create_sensor_dropdown():
    """Create the sensor dropdown component"""
    locations = get_unique_locations()
    return dcc.Dropdown(
        id="sensor-dropdown",
        options=[{"label": location, "value": location} for location in locations],
        value=locations[:2] if len(locations) >= 2 else locations,
        multi=True
    )

# Create the dropdown components
device_dropdown = create_device_dropdown()
sensor_dropdown = create_sensor_dropdown()
