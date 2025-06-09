from dash import Input, Output, State
from utils.data_loader import load_data
from components.radar_graphs import update_radar_graph
from components.overlay_plot import create_overlay_figure
from components.signal_graphs import update_signal_graphs

def register_callbacks(app):
    @app.callback(
        [Output("ratings-dropdown-container", "style"),
         Output("sensor-dropdown-container", "style")],
        Input("view-tabs", "value")
    )
    def toggle_dropdown_visibility(tab):
        if tab == "ratings":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}

    @app.callback(
        Output("rating-health-graph", "figure"),
        [
            Input("start-date", "date"),
            Input("end-date", "date"),
            Input("start-time", "value"),
            Input("end-time", "value"),
            Input("device-dropdown", "value"),
            Input("ratings-dropdown", "value")
        ],
        State("view-tabs", "value")
    )
    def update_health_graph(start_date, end_date, start_time, end_time, selected_device, selected_ratings, current_tab):
        if current_tab != "ratings" or selected_ratings is None or selected_device is None:
            from plotly.graph_objects import Figure
            return Figure()
        
        start_datetime = f"{start_date} {start_time}" if start_date and start_time else None
        end_datetime = f"{end_date} {end_time}" if end_date and end_time else None
        
        df = load_data(selected_device, start_datetime, end_datetime)
        df = df[df["Device"] == selected_device]
        
        return create_overlay_figure(df, y_columns=selected_ratings, y_label="Rating Health")

    @app.callback(
        [
            Output("timestamp-header", "children"),
            Output("radar-graph-1", "figure"),
            Output("radar-graph-2", "figure"),
            Output("frequency-graph", "figure"),
            Output("signal-graph-sig-raw", "figure"),
            Output("signal-graph-sig-fft", "figure"),
            Output("signal-graph-env", "figure"),
            Output("signal-graph-env-fft", "figure")
        ],
        [
            Input("start-date", "date"),
            Input("end-date", "date"),
            Input("start-time", "value"),
            Input("end-time", "value"),
            Input("device-dropdown", "value"),
            Input("sensor-dropdown", "value"),
            Input("view-tabs", "value")
        ]
    )
    def update_all_graphs(start_date, end_date, start_time, end_time, selected_device, selected_locations, current_tab):
        if selected_locations is None or selected_device is None:
            from plotly.graph_objects import Figure
            empty_fig = Figure()
            return "No data loaded", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
        start_datetime = f"{start_date} {start_time}" if start_date and start_time else None
        end_datetime = f"{end_date} {end_time}" if end_date and end_time else None

        df = load_data(selected_device, start_datetime, end_datetime)
        df = df[df["Device"] == selected_device]
        
        if current_tab == "signal":
            df = df[df["location"].isin(selected_locations)]

        if df.empty or "timestamp" not in df.columns:
            header_str = "No data loaded"
        else:
            start = df["timestamp"].min().strftime('%Y-%m-%d %H:%M:%S')
            end = df["timestamp"].max().strftime('%Y-%m-%d %H:%M:%S')
            header_str = f"{start} - {end}"

        radar1 = update_radar_graph(df, graph_id=1)
        radar2 = update_radar_graph(df, graph_id=2)
        freq_fig = create_overlay_figure(df, y_columns=["High-Frequency Acceleration"], y_label="High-Frequency Acceleration (a.u.)")
        signal_figs = update_signal_graphs(df)

        return header_str, radar1, radar2, freq_fig, *signal_figs

    @app.callback(
        Output("ratings-view", "style"),
        Output("signal-view", "style"),
        Input("view-tabs", "value")
    )
    def toggle_tab_view(tab):
        if tab == "ratings":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}