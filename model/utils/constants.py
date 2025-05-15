INPUT_FEATURES = ['Low-Frequency Acceleration Z','High-Frequency Acceleration', 'Temperature', 'Vibration Velocity Z']

DEVICE_TARGET_FEATURES = {
    "conveyor_belt": [
        'alignment_status', 'bearing_lubrication', 'crest_factor', 'electromagnetic_status',
        'fit_condition', 'kurtosis_opt', 'rms_10_25khz', 'rms_1_10khz',
        'rotor_balance_status', 'rubbing_condition', 'velocity_rms', 'peak_value_opt'
    ],
    "high_temp_fan": ['velocity_rms', 'crest_factor', 'kurtosis_opt', 'rms_1_10khz',
       'rms_10_25khz', 'rotor_balance_status', 'alignment_status',
       'fit_condition', 'bearing_lubrication', 'rubbing_condition',
       'electromagnetic_status'
     ],
    "tube_mill": ['velocity_rms', 'crest_factor', 'kurtosis_opt', 'rms_1_10khz',
       'rms_10_25khz', 'peak_value_opt', 'rms_0_10hz', 'rms_10_100hz',
       'peak_10_1000hz', 'rotor_balance_status', 'alignment_status',
       'fit_condition', 'bearing_lubrication', 'rubbing_condition',
       'electromagnetic_status'
    ]
}