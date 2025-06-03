import argparse
from flask import Flask, redirect
import dash
from dash import html
import dash_bootstrap_components as dbc
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument parsing
parser = argparse.ArgumentParser(description="Run dashboard app")
parser.add_argument('--aws', action='store_true', help='Load data from S3 bucket instead of local file')
args, unknown = parser.parse_known_args()

# Set environment variable for AWS mode before importing data_loader
if args.aws:
    os.environ['DASHBOARD_AWS_MODE'] = 'true'
    logger.info("AWS mode enabled via environment variable")

# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app with Flask server
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    url_base_pathname='/dashboard/'
)
app.title = "Device Sensor Dashboard"

# Now import components after AWS mode is set
from utils.data_loader import data_loader
from components.layout import create_layout
from callbacks.handlers import register_callbacks

# Set up the Dash layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

@server.route('/')
def index():
    """Redirect root to dashboard"""
    return redirect('/dashboard/')

if __name__ == "__main__":
    # Initial data load
    logger.info("Performing initial data load...")
    data_loader.update_all_data()
    
    logger.info("Starting Flask server...")
    server.run(debug=True, host='0.0.0.0', port=8050)

    # Uncomment to run HMR for development
    # logger.info("Starting Dash server...")
    # app.run_server(debug=True, host='0.0.0.0', port=8050)