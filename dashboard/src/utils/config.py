# techdebt: hardcoding col names - should vary by what is in df
RATINGS = {
    "status_cols": [
        "alignment_status", "bearing_lubrication", "electromagnetic_status",
        "fit_condition", "rotor_balance_status", "rubbing_condition"
    ], 
    "metric_cols": [
        "velocity_rms", "crest_factor", "kurtosis_opt",
        "peak_value_opt", "rms_10_25khz", "rms_1_10khz"
    ]}

# TODO: update descriptions with final models
RATING_DESCRIPTIONS = {
    "velocity_rms": "√(1/N × Σ(v_z²))",
    "crest_factor": "Peak / RMS",
    "kurtosis_opt": "μ₄ / μ₂²",
    "peak_value_opt": "max(|x|)",
    "rms_10_25khz": "√(1/N × Σ(x²)) in 10–25.6 kHz band",
    "rms_1_10khz": "√(1/N × Σ(x²)) in 1–10 kHz band",
    "rms_0.1_10hz": "√(1/N × Σ(x²)) in 0.1–10 Hz band",
    "rms_10_100hz": "√(1/N × Σ(x²)) in 10–100 Hz band",
    "peak_10_1000hz": "max(|x|) in 10–1000 Hz band",
    "alignment_status": "f(axial_vibration_pattern)",
    "bearing_lubrication": "f(bearing_freq_energy)",
    "electromagnetic_status": "f(motor_current_harmonics)",
    "fit_condition": "f(impulse_signal_features)",
    "rotor_balance_status": "f(phase_diff, amplitude)",
    "rubbing_condition": "f(high_freq_noise, friction)"
}

# TODO: add colours for other devices' locations
LOCATION_COLOUR_MAP = {
    "Gear Reducer": "#1f77b4",
    "Gearbox First Shaft Input End": "#ff7f0e",
    "Motor Drive End": "#9467bd",
}

LOCATION_COLOUR_EMOJI = {
    "Gear Reducer": "🟦",
    "Gearbox First Shaft Input End": "🟧",
    "Motor Drive End": "🟪",
}

RATING_COLOUR_MAP = {
    "alignment_status": "#e74c3c",
    "bearing_lubrication": "#3498db",
    "electromagnetic_status": "#2ecc71",
    "fit_condition": "#f39c12",
    "rotor_balance_status": "#a861c4",
    "rubbing_condition": "#743f10",
    "velocity_rms": "#d1c719",
    "crest_factor": "#34495e",
    "kurtosis_opt": "#16a085",
    "peak_value_opt": "#d35400",
    "rms_10_25khz": "#5e44ad",
    "rms_1_10khz": "#20894c",
}

RATING_COLOUR_EMOJI = {
    "alignment_status": "🔴",
    "bearing_lubrication": "🔵",
    "electromagnetic_status": "🟢",
    "fit_condition": "🟠",
    "rotor_balance_status": "🟣",
    "rubbing_condition": "🟤",
    "velocity_rms": "🟡",
    "crest_factor": "⚫",
    "kurtosis_opt": "🔵🟢",
    "peak_value_opt": "🟠🔴",
    "rms_10_25khz": "🟣🔵",
    "rms_1_10khz": "⚫🟢",
}

# techdebt: add other feature col names
FEATURES = {"vibration_velocity_z":"Vibration Velocity Z"}