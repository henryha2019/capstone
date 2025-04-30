from dash import Input, Output
from utils.data_loader import load_data
from components.radar_chart import update_radar_chart
from components.overlay_plot import update_frequency_chart
from components.line_charts import update_detail_charts


def register_callbacks(app):
    @app.callback(
        [
            Output("radar-chart-1", "figure"),
            Output("radar-chart-2", "figure"),
            Output("frequency-chart", "figure"),
            Output("detail-chart-1", "figure"),
            Output("detail-chart-2", "figure"),
            Output("detail-chart-3", "figure"),
            Output("detail-chart-4", "figure")
        ],
        [
            Input("device-dropdown", "value"),
            Input("sensor-dropdown", "value")
        ]
    )
    def update_all_charts(selected_device, selected_sensors):
        df = load_data(selected_device, selected_sensors)
        
        radar1 = update_radar_chart(df, chart_id=1)
        radar2 = update_radar_chart(df, chart_id=2)
        freq_fig = update_frequency_chart(df)
        detail_figs = update_detail_charts(df)

        return radar1, radar2, freq_fig, *detail_figs
