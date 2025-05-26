from dash import html
from utils.data_loader import load_data

def header_timestamp():
    "Returns a string showing the time range of the data in the dashboard to display in the dashboard header"
    return html.H1(id='timestamp-header', className="dashboard-header")
