#!/usr/bin/env python3
# compute_all_metrics.py
#
# Walks through every *.Single under Data/voltage/extracted/,
# reads each waveform (4-byte float32 header + float32 samples),
# and computes:
#   • velocity_rms
#   • crest_factor
#   • kurtosis_opt
#   • peak_value_opt
#   • rms_0_10hz
#   • rms_10_100hz
#   • rms_1_10khz
#   • rms_10_25khz
#   • peak_10_1000hz
#
# It also creates placeholder columns for:
#   rotor_balance_status, alignment_status, fit_condition,
#   bearing_lubrication, rubbing_condition, electromagnetic_status
#
# Finally, it prints the first 10 rows and saves a CSV to Data/voltage/metrics_all.csv
# --------------------------------------------------------------------------- #

import os
import re
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats, signal

# —————————— 1. CONFIGURATION ——————————
# This script assumes it lives at <project_root>/model/src/compute_all_metrics.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "Data" / "voltage"
EXTRACT_DIR  = DATA_DIR / "extracted"

# Sampling rate is read from each .Single’s header, so no fixed fs here.

# Status columns (initially <NA>)
STATUS_COLS = [
    "rotor_balance_status",
    "alignment_status",
    "fit_condition",
    "bearing_lubrication",
    "rubbing_condition",
    "electromagnetic_status",
]

# Metric column names
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

# —————————— 2. HELPERS: Read waveform from a .Single file ——————————
def read_single_file(path: Path) -> (float, np.ndarray):
    """
    Read a .Single file as binary. Return (sampling_rate_hz, signal_array).
    Format:
      - First 4 bytes: little-endian float32 = sampling rate (Hz)
      - Remainder: consecutive float32 samples (waveform)
    """
    with open(path, "rb") as f:
        fs = struct.unpack("<f", f.read(4))[0]  # 4-byte header
        sig = np.fromfile(f, dtype="<f4")      # rest of file = float32 samples
    return fs, sig.flatten()

# —————————— 3. HELPERS: DSP metrics ——————————
def band_rms(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """
    Compute RMS of `sig` in frequency band [f_lo, f_hi] via FFT integration.
    """
    n     = sig.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    X     = np.fft.rfft(sig - sig.mean())
    power = (np.abs(X) ** 2) / n
    mask  = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return np.nan
    return np.sqrt(power[mask].sum())

def band_peak(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """
    Compute peak absolute amplitude of `sig` filtered to [f_lo, f_hi] via FFT mask.
    """
    n     = sig.size
    X     = np.fft.rfft(sig - sig.mean())
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask  = (freqs >= f_lo) & (freqs <= f_hi)
    X_f   = np.zeros_like(X)
    X_f[mask] = X[mask]
    x_f   = np.fft.irfft(X_f, n)
    return np.abs(x_f).max()

# —————————— 4. FIND all .Single files and initialize records ——————————
records = []

for subdir in EXTRACT_DIR.iterdir():
    if not subdir.is_dir():
        continue
    for root, _, files in os.walk(subdir):
        for fname in files:
            if not fname.lower().endswith(".single"):
                continue

            single_path = Path(root) / fname

            # Parse timestamp from filename: ..._YYYYMMDD_HHMMSS.Single
            m = re.search(r"_(\d{8})_(\d{6})\.Single$", fname, flags=re.IGNORECASE)
            if not m:
                continue
            date_str, time_str = m.groups()
            ts = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

            # Prepare an empty record
            rec = {"timestamp": ts, "filepath": str(single_path.relative_to(PROJECT_ROOT))}
            # Pre‐populate all metric & status columns with NA
            for col in METRIC_COLS + STATUS_COLS:
                rec[col] = pd.NA

            records.append(rec)

# Sort records by timestamp
records.sort(key=lambda r: r["timestamp"])

# Build DataFrame
df = pd.DataFrame(records)

# —————————— 5. LOOP over each record, load waveform, compute metrics ——————————
for idx, row in df.iterrows():
    path = PROJECT_ROOT / row["filepath"]
    try:
        fs, wave = read_single_file(path)
    except Exception as e:
        print(f"⚠ Skipping '{row['filepath']}' (could not read): {e}")
        continue

    # 1. velocity_rms
    vel_rms = np.sqrt(np.mean(wave**2))

    # 2. crest_factor
    peak_val = np.max(np.abs(wave))
    crest    = peak_val / vel_rms if vel_rms > 0 else np.nan

    # 3. kurtosis_opt
    kurt = stats.kurtosis(wave, fisher=False)

    # 4. peak_value_opt
    peak_value = peak_val

    # 5. rms_0_10hz
    rms_0_10   = band_rms(wave, fs, 0.1,   10)

    # 6. rms_10_100hz
    rms_10_100 = band_rms(wave, fs, 10,   100)

    # 7. rms_1_10khz
    rms_1_10k  = band_rms(wave, fs, 1_000, 10_000)

    # 8. rms_10_25khz
    rms_10_25k = band_rms(wave, fs, 10_000, 25_000)

    # 9. peak_10_1000hz
    peak_10_1000 = band_peak(wave, fs, 10, 1_000)

    # ASSIGN into DataFrame
    df.at[idx, "velocity_rms"]      = vel_rms
    df.at[idx, "crest_factor"]      = crest
    df.at[idx, "kurtosis_opt"]      = kurt
    df.at[idx, "peak_value_opt"]    = peak_value
    df.at[idx, "rms_0_10hz"]        = rms_0_10
    df.at[idx, "rms_10_100hz"]      = rms_10_100
    df.at[idx, "rms_1_10khz"]       = rms_1_10k
    df.at[idx, "rms_10_25khz"]      = rms_10_25k
    df.at[idx, "peak_10_1000hz"]    = peak_10_1000

    # STATUS FIELDS remain <NA> (fill these later as needed)
    # df.at[idx, "rotor_balance_status"]   = ...
    # df.at[idx, "alignment_status"]       = ...
    # df.at[idx, "fit_condition"]          = ...
    # df.at[idx, "bearing_lubrication"]    = ...
    # df.at[idx, "rubbing_condition"]      = ...
    # df.at[idx, "electromagnetic_status"] = ...

# —————————— 6. DISPLAY & SAVE ——————————
print("\n— First 10 rows of all metrics —")
print(df.head(10).to_string(index=False))

# Save to CSV
out_csv = DATA_DIR / "metrics_all.csv"
df.to_csv(out_csv, index=False)
print(f"\nFull table saved to: {out_csv}")
