def format_plot(fig):
    return fig.update_layout(
        title=None,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False
    )
