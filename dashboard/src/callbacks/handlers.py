from dash import Input, Output
from utils.data_loader import load_data
from components.radar_chart import update_radar_chart
from components.overlay_plot import update_frequency_chart
from components.signal_charts import update_signal_charts

def register_callbacks(app):
    @app.callback(
        [
            Output("radar-chart-1", "figure"),
            Output("radar-chart-2", "figure"),
            Output("frequency-chart", "figure"),
            Output("signal-chart-sig-raw", "figure"),
            Output("signal-chart-sig-fft", "figure"),
            Output("signal-chart-env", "figure"),
            Output("signal-chart-env-fft", "figure")
        ],
        [
            Input("device-dropdown", "value"),
            Input("sensor-dropdown", "value")
        ]
    )
    def update_all_charts(selected_device, selected_sensors):
        # todo: if no sensor selected, disappear plots and give message
        # if not selected_sensors:
        
        df = load_data()
        df = df[(df["Device"] == selected_device) & (df["location"].isin(selected_sensors))]
        
        radar1 = update_radar_chart(df, chart_id=1)
        radar2 = update_radar_chart(df, chart_id=2)
        # techdebt: rename freq to overlay
        freq_fig = update_frequency_chart(df)
        signal_figs = update_signal_charts(df)

        return radar1, radar2, freq_fig, *signal_figs
