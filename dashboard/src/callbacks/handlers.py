from dash import Input, Output
from utils.data_loader import load_data
from components.radar_chart import update_radar_chart
from components.overlay_plot import update_frequency_chart
from components.signal_charts import update_signal_charts

def register_callbacks(app):
    @app.callback(
        [
            Output("timestamp-header", "children"),
            Output("radar-chart-1", "figure"),
            Output("radar-chart-2", "figure"),
            Output("frequency-chart", "figure"),
            Output("signal-chart-sig-raw", "figure"),
            Output("signal-chart-sig-fft", "figure"),
            Output("signal-chart-env", "figure"),
            Output("signal-chart-env-fft", "figure")
        ],
        [
            Input("date-range-dropdown", "value"),
            Input("device-dropdown", "value"),
            Input("sensor-dropdown", "value")
        ]
    )
    def update_all_charts(selected_time_range, selected_device, selected_sensors):
        # todo: if no sensor selected, disappear plots and give message
        # if not selected_sensors:
        
        df = load_data(selected_time_range)
        df = df[(df["Device"] == selected_device) & (df["location"].isin(selected_sensors))]
        
        if df.empty or "timestamp" not in df.columns:
            header_str = "No data loaded"
        else:
            start = df["timestamp"].min().strftime('%Y-%m-%d %H:%M:%S')
            end = df["timestamp"].max().strftime('%Y-%m-%d %H:%M:%S')
            header_str = f"{start} - {end}"

        radar1 = update_radar_chart(df, chart_id=1)
        radar2 = update_radar_chart(df, chart_id=2)
        # techdebt: rename freq to overlay
        freq_fig = update_frequency_chart(df)
        signal_figs = update_signal_charts(df)

        return header_str, radar1, radar2, freq_fig, *signal_figs

    @app.callback(
        Output("summary-view", "style"),
        Output("signal-view", "style"),
        Input("view-tabs", "value")
    )
    def toggle_tab_view(tab):
        if tab == "summary":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}
