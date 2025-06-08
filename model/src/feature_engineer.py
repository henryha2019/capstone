#!/usr/bin/env python3
# feature_eng.py
#
# This script will:
#  1. Find all .json files under each “<date>#Belt Conveyer” folder inside Data/voltage/
#  2. Read axisX/axisY from each JSON
#  3. Compute fs = 1 / (axisX[1] - axisX[0])
#  4. Treat axisY as the raw waveform
#  5. Compute DSP metrics (velocity_rms, crest_factor, etc.)
#  6. Save metrics to Data/voltage/metrics_json.csv
#  7. Read merged ratings/features from Data/process/8#Belt Conveyer_merged.csv
#  8. Rename all 12 rating columns (append "_rating")
#  9. Bucket and summarize measurement features per rating/time window
# 10. Join JSON metrics + bucket summary on matching location & time
# 11. Save full features to Data/process/8#Belt Conveyer_full_features.csv
# ---------------------------------------------------------------------------- #

import os
import re
import json
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from scipy import stats

# —————————— 1. CONFIGURATION ——————————
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOLTAGE_DIR  = PROJECT_ROOT / "Data" / "voltage"
PROCESS_DIR  = PROJECT_ROOT / "Data" / "process"

# DSP metrics to compute from JSON
METRIC_COLS = [
    "velocity_rms",
    "crest_factor",
    "kurtosis_opt",
    "peak_value_opt",
    "rms_0_10hz",
    "rms_10_100hz",
    "rms_1_10khz",
    "rms_10_25khz",
    "peak_10_1000hz",
]

# All 12 rating columns in merged CSV
RATING_COLS = [
    "alignment_status",
    "bearing_lubrication",
    "crest_factor",
    "electromagnetic_status",
    "fit_condition",
    "kurtosis_opt",
    "rms_10_25khz",
    "rms_1_10khz",
    "rotor_balance_status",
    "rubbing_condition",
    "velocity_rms",
    "peak_value_opt",
]

# Map from sensor_id hex to human-readable location
SENSOR_MAP = {
    "67c29baa30e6dd385f031b30": "Motor Non-Drive End",
    "67c29baa30e6dd385f031b39": "Motor Drive End",
    "67c29baa30e6dd385f031b42": "Fan Free End",
    "67c29baa30e6dd385f031b4b": "Fan Inlet End",
    "67c29bab30e6dd385f031c98": "Motor Drive End",
    "67c29bab30e6dd385f031ca1": "Gearbox First Shaft Input End",
    "67c29bab30e6dd385f031caa": "Gear Reducer",
    "67c29bab30e6dd385f031c2c": "Motor Non-Drive End",
    "67c29bab30e6dd385f031c35": "Motor Drive End",
    "67c29bab30e6dd385f031c3e": "Gearbox First Shaft Input End",
    "67c29bab30e6dd385f031c47": "Left-side Bearing Housing of Gear Set",
    "67c29bab30e6dd385f031c50": "Right-side Bearing Housing of Gear Set",
}

# Regex to parse file names: capture sensor_id & wave_code
FILENAME_PATTERN = re.compile(
    r"^\d{8} \d{6}_[0-9a-f]{24}_([0-9a-f]{24})_([A-Z]{2})\.json$"
)

# —————————— 2. HELPERS: JSON reader & DSP metrics ——————————
def read_json_file(path: Path) -> (float, np.ndarray):
    with open(path, "r") as f:
        data = json.load(f)
    axis_x = np.array(data["axisX"], dtype=np.float64)
    axis_y = np.array(data["axisY"], dtype=np.float32)
    if len(axis_x) < 2:
        raise ValueError(f"Not enough points in axisX of {path.name}")
    dt = float(axis_x[1] - axis_x[0])
    if dt <= 0:
        raise ValueError(f"Invalid dt ({dt}) in {path.name}")
    return 1.0 / dt, axis_y


def band_rms(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    n     = sig.size
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    X     = np.fft.rfft(sig - sig.mean())
    power = (np.abs(X)**2) / n
    mask  = (freqs >= f_lo) & (freqs <= f_hi)
    return np.sqrt(power[mask].sum()) if np.any(mask) else np.nan


def band_peak(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    n     = sig.size
    X     = np.fft.rfft(sig - sig.mean())
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mask  = (freqs >= f_lo) & (freqs <= f_hi)
    X_f   = np.zeros_like(X); X_f[mask] = X[mask]
    return np.abs(np.fft.irfft(X_f, n)).max()

# —————————— 3. BUCKET SUMMARY FUNCTION ——————————
def bucket_summary(
    df: pd.DataFrame,
    measurement_cols: list[str],
    rating_cols: list[str],
    time_col: str = 'datetime',
    location_col: str = 'location',
    bucket_minutes: int = 20
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([location_col, time_col]).reset_index(drop=True)

    bucket_ids = [None] * len(df)
    counter = 0

    for loc, grp in df.groupby(location_col, sort=False):
        current_id = None
        start_time = None
        current_rt  = None
        for idx in grp.index:
            rtup = tuple(df.at[idx, c] for c in rating_cols)
            t    = df.at[idx, time_col]
            if current_id is None:
                counter += 1
                current_id  = counter
                start_time  = t
                current_rt   = rtup
            else:
                if rtup != current_rt or (t - start_time) >= pd.Timedelta(minutes=bucket_minutes):
                    counter += 1
                    current_id  = counter
                    start_time  = t
                    current_rt   = rtup
            bucket_ids[idx] = current_id

    df['bucket_id'] = bucket_ids

    agg = {}
    for m in measurement_cols:
        agg[f"{m}_count"] = (m, 'count')
        agg[f"{m}_mean"]  = (m, 'mean')
        agg[f"{m}_std"]   = (m, 'std')
        agg[f"{m}_min"]   = (m, 'min')
        agg[f"{m}_max"]   = (m, 'max')
    for r in rating_cols:
        agg[f"{r}"] = (r, 'first')

    agg['bucket_start'] = (time_col, 'min')
    agg['bucket_end']   = (time_col, 'max')

    return (
        df
        .groupby([location_col, 'bucket_id'])
        .agg(**agg)
        .reset_index()
    )

# —————————— 4. JSON METRICS COLLECTION ——————————
records = []
for subdir in VOLTAGE_DIR.iterdir():
    if not subdir.is_dir() or "Belt Conveyer" not in subdir.name:
        continue
    for root, _, files in os.walk(subdir):
        for fn in files:
            if not fn.lower().endswith('.json'):
                continue
            # parse timestamp from filename
            m = (
                re.search(r"(\d{8})\s?(\d{6})_", fn)
                or re.search(r"_(\d{8})_(\d{6})\.json$", fn)
            )
            if not m:
                continue
            d, t = m.groups()
            ts   = datetime.strptime(d + t, '%Y%m%d%H%M%S')

            rec = {"timestamp": ts, "filepath": str(Path(root)/fn).split(str(PROJECT_ROOT)+os.sep)[1]}
            for col in METRIC_COLS:
                rec[col] = pd.NA
            records.append(rec)

# build dataframe
records.sort(key=lambda r: r['timestamp'])
metrics_df = pd.DataFrame(records)

# --- EXTRACT SENSOR ID, WAVE CODE & LOCATION from file name ---
metrics_df['file_name'] = metrics_df['filepath'].apply(os.path.basename)
extracted = metrics_df['file_name'].str.extract(FILENAME_PATTERN)
metrics_df['sensor_id'] = extracted[0]
metrics_df['wave_code'] = extracted[1]
metrics_df['location']  = metrics_df['sensor_id'].map(SENSOR_MAP)
metrics_df.drop(columns=['file_name'], inplace=True)

# compute DSP metrics
for i, row in metrics_df.iterrows():
    p = PROJECT_ROOT / row['filepath']
    try:
        fs, w = read_json_file(p)
    except Exception:
        continue

    vrs = np.sqrt(np.mean(w**2))
    pk  = np.max(np.abs(w))
    vals = [
        vrs,
        pk / vrs if vrs > 0 else np.nan,
        stats.kurtosis(w, fisher=False),
        pk,
        band_rms(w, fs,   0.1,   10),
        band_rms(w, fs,    10,  100),
        band_rms(w, fs,  1000,10000),
        band_rms(w, fs, 10000,25000),
        band_peak(w, fs,   10, 1000),
    ]
    for j, col in enumerate(METRIC_COLS):
        metrics_df.at[i, col] = vals[j]

# save JSON metrics
out_json = VOLTAGE_DIR / "metrics_json.csv"
metrics_df.to_csv(out_json, index=False)
print(f"Saved metrics to {out_json}")

# —————————— 5. LOAD MERGED & RENAME RATINGS ——————————
merged_path = PROCESS_DIR / "8#Belt Conveyer_merged.csv"
merged_df   = pd.read_csv(merged_path, parse_dates=["datetime"])

# rename all 12 ratings by appending '_rating'
merged_df = merged_df.rename(columns={c: f"{c}_rating" for c in RATING_COLS})

# —————————— 6. BUCKET SUMMARY ON MEASUREMENTS ——————————
measurement_cols = [
    'High-Frequency Acceleration',
    'Low-Frequency Acceleration Z',
    'Temperature',
    'Vibration Velocity Z'
]
rating_cols_renamed = [f"{c}_rating" for c in RATING_COLS]

summary_df = bucket_summary(
    merged_df,
    measurement_cols=measurement_cols,
    rating_cols=rating_cols_renamed,
    time_col='datetime',
    location_col='location',
    bucket_minutes=20
)

# use bucket start as the merge key
summary_df = summary_df.rename(columns={'bucket_start': 'datetime'})
# save bucket summary if desired
out_bucket = PROCESS_DIR / "8#Belt Conveyer_bucket_summary.csv"
summary_df.to_csv(out_bucket, index=False)
print(f"Saved bucket summary to {out_bucket}")

# —————————— 7. MERGE METRICS + SUMMARY VIA INTERVAL-INDEX LOOKUP ——————————

# 1) Rename & convert metrics timestamp to datetime
metrics_df = metrics_df.rename(columns={'timestamp': 'datetime'})
metrics_df['datetime'] = pd.to_datetime(metrics_df['datetime'])

# 2) Make sure summary_df has datetime=bucket_start and bucket_end as Timestamps
summary_df['datetime']   = pd.to_datetime(summary_df['datetime'])
summary_df['bucket_end'] = pd.to_datetime(summary_df['bucket_end'])

# 3) Build an IntervalIndex for each bucket
intervals = pd.IntervalIndex.from_arrays(
    summary_df['datetime'],    # bucket_start
    summary_df['bucket_end'],
    closed='both'
)

# 4) Attach that interval to each summary row and index by (location, interval)
summary_idx = (
    summary_df
    .assign(interval=intervals)
    .set_index(['location','interval'])
)

# 5) For each metric row, look up the bucket whose interval contains its timestamp
matched = []
for _, row in metrics_df.iterrows():
    loc = row['location']
    ts  = row['datetime']
    try:
        bucket = summary_idx.loc[(loc, ts)]
    except KeyError:
        # no bucket covers this timestamp
        continue
    # combine metric + bucket fields into one dict
    combined = {**row.to_dict(), **bucket.to_dict()}
    matched.append(combined)

# 6) Build final DataFrame and save
full_df = pd.DataFrame(matched)
out_full = PROCESS_DIR / "8#Belt Conveyer_full_features.csv"
full_df.to_csv(out_full, index=False)
print(f"Saved full features to {out_full}")


