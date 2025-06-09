from dash import html
from utils.data_loader import load_data

def header_timestamp():
    "Returns a string showing the selected time range of the data"
    return html.H1(id='timestamp-header', className="dashboard-header")
