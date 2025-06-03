from dash import dcc
import plotly.graph_objects as go
from utils.plot_config import format_plot
from components.ratings_descriptions import DESCRIPTIONS

# techdebt: more descriptive names
radar_chart_1 = dcc.Graph(
    id="radar-chart-1", 
    figure=go.Figure(), 
    className="graph-container", 
    config={"displayModeBar": False}
    )

# techdebt: more descriptive names
radar_chart_2 = dcc.Graph(
    id="radar-chart-2", 
    figure=go.Figure(), 
    className="graph-container", 
    config={"displayModeBar": False}
    )

# techdebt: hardcoding col names - should vary by what is in df
CHART_1_COLS = [
    "alignment_status", "bearing_lubrication", "electromagnetic_status",
    "fit_condition", "rotor_balance_status", "rubbing_condition"
]

# techdebt: hardcoding col names - should include all remaining, and narrow down without breaking
CHART_2_COLS = [
    "velocity_rms", "crest_factor", "kurtosis_opt",
    "peak_value_opt", "rms_10_25khz", "rms_1_10khz"
]

def update_radar_chart(df, chart_id):
    if df.empty:
        return go.Figure()
    
    cols = CHART_1_COLS if chart_id == 1 else CHART_2_COLS
    means = df[cols].mean()
    labels = [
        f"<span style='text-align:center; display:block'>{col.replace('_', ' ').title()}<br>{v:.1f}</span>"
        for col, v in zip(means.index, means.values)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=means.values,
        theta=labels,
        fill="toself",
        name="",
        hovertemplate=[
        f"{DESCRIPTIONS.get(col, '')}"
        for col in means.index
        ],
        hoverlabel=dict(
            namelength=0,
            font=dict(size=18)
        )
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(tickfont=dict(size=16))
        ),
        showlegend=False,
        margin=dict(t=10, l=10, r=10, b=10)
    )

    return format_plot(fig)
