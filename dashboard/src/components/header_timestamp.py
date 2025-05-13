from dash import html
from utils.data_loader import load_data

def header_timestamp():
    "Returns a string showing the time range of the data in the dashboard to display in the dashboard header"
    df = load_data()
    if df.empty or "timestamp" not in df.columns:
        timestamp_str = "No data loaded"
    else:
        earliest_timestamp = df["timestamp"].min()
        earliest_timestamp_str = earliest_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        latest_timestamp = df["timestamp"].max()
        latest_timestamp_str = latest_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        timestamp_str = f"{earliest_timestamp_str} - {latest_timestamp_str}"

    return html.H1(timestamp_str, className="dashboard-header")
