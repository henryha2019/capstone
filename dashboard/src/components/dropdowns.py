from dash import dcc, html
import dash_bootstrap_components as dbc
from utils.data_loader import get_unique_devices, get_unique_locations, data_loader
from utils.config import LOCATION_COLOUR_EMOJI, RATING_COLOUR_EMOJI, RATINGS

def create_device_dropdown():
    devices = get_unique_devices()
    return dcc.Dropdown(
        id="device-dropdown",
        options=[{"label": device, "value": device} for device in devices],
        value=devices[0] if devices else None,
        clearable=False
    )

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

def create_sub_device_dropdown(options=None, default_values=None, dropdown_id="sensor-dropdown"):
    if options is None:
        locations = get_unique_locations()
        options = [{"label": f"{LOCATION_COLOUR_EMOJI.get(loc, '')} {loc}", "value": loc} for loc in locations]
        default_values = locations if default_values is None else default_values
    else:
        if default_values is None:
            default_values = [options[0]["value"]] if options else []
    
    return dcc.Dropdown(
        id=dropdown_id,
        options=options,
        value=default_values,
        multi=True,
        placeholder=""
    )

def create_ratings_dropdown():
    ratings = RATINGS["status_cols"] + RATINGS["metric_cols"]
    options = [
        {
            "label": f"{RATING_COLOUR_EMOJI.get(r, '')} {r.replace('_', ' ').title()}", 
            "value": r
        } 
        for r in ratings
    ]
    
    return create_sub_device_dropdown(
        options=options,
        default_values=ratings,
        dropdown_id="ratings-dropdown"
    )

def create_sensor_dropdown():
    locations = get_unique_locations()
    options = [{"label": f"{LOCATION_COLOUR_EMOJI.get(loc, '')} {loc}", "value": loc} for loc in locations]
    
    return create_sub_device_dropdown(
        options=options,
        default_values=locations,
        dropdown_id="sensor-dropdown"
    )

def create_dual_dropdown():
    return html.Div([
        # Ratings dropdown
        html.Div(
            create_ratings_dropdown(),
            id="ratings-dropdown-container",
            style={"display": "block"}
        ),
        
        # Locations dropdown
        html.Div(
            create_sensor_dropdown(),
            id="sensor-dropdown-container", 
            style={"display": "none"}
        )
    ])

device_dropdown = create_device_dropdown()
dual_dropdown = create_dual_dropdown()