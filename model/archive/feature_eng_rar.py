#!/usr/bin/env python3
"""
compute_all_metrics.py

1. Unpacks all .rar archives in data/rar_files (in place) using patool
2. Walks through every *.Single under data/rar_files,
   reads each waveform (4-byte float32 header + float32 samples),
   computes DSP metrics + placeholder status cols,
   and plots each signal (time vs. amplitude) into docs/images/rar_plots/
3. Prints first 10 rows and saves CSV to data/rar_files/metrics_all.csv
"""

import struct
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import patoolib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAR_DIR      = PROJECT_ROOT / "data" / "rar_files"
PLOT_DIR     = PROJECT_ROOT / "docs" / "images" / "rar_plots"
OUT_CSV      = RAR_DIR   / "metrics_all.csv"

STATUS_COLS = [
    "rotor_balance_status",
    "alignment_status",
    "fit_condition",
    "bearing_lubrication",
    "rubbing_condition",
    "electromagnetic_status",
]

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

TIMESTAMP_PATTERN = re.compile(r"_(\d{8})_(\d{6})\.Single$", re.IGNORECASE)


def ensure_directories():
    """
    Ensure that the RAR_DIR and PLOT_DIR directories exist.

    This function creates the directories for raw `.rar` files and output plots
    if they do not already exist.

    Args:
        None

    Returns:
        None

    Example:
        ensure_directories()
    """
    RAR_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def unpack_archives():
    """
    Extract all .rar files in RAR_DIR in place using patoolib.

    This function locates each `.rar` archive in RAR_DIR and extracts its
    contents back into the same directory. For each archive, it prints
    a success or failure message.

    Args:
        None

    Returns:
        None

    Example:
        unpack_archives()
    """
    for rar in RAR_DIR.glob("*.rar"):
        try:
            patoolib.extract_archive(str(rar), outdir=str(RAR_DIR))
            print(f"✔ Extracted {rar.name} via patoolib")
        except Exception as e:
            print(f"⚠ patoolib failed on {rar.name}: {e}")


def read_single_file(path: Path):
    """
    Read a .Single waveform file and return its sampling rate and samples.

    This function opens the given `.Single` file in binary mode, reads the
    first 4 bytes as a little-endian float32 sampling rate, and then reads
    the remainder as an array of float32 samples.

    Args:
        path (Path): Path to the `.Single` file.

    Returns:
        tuple:
            float: Sampling rate in Hz.
            np.ndarray: 1D array of waveform samples.

    Example:
        fs, wave = read_single_file(Path("data/rar_files/device_20250101_000000.Single"))
    """
    with open(path, "rb") as f:
        fs  = struct.unpack("<f", f.read(4))[0]
        sig = np.fromfile(f, dtype="<f4")
    return fs, sig.flatten()


def band_rms(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """
    Compute the RMS of a signal in a given frequency band via FFT integration.

    This function subtracts the mean, computes the FFT, and integrates the
    power spectral density between f_lo and f_hi to compute the band-limited RMS.

    Args:
        sig (np.ndarray): Time-domain signal.
        fs (float): Sampling rate in Hz.
        f_lo (float): Lower bound of frequency band (Hz).
        f_hi (float): Upper bound of frequency band (Hz).

    Returns:
        float: Band-limited RMS value, or NaN if the band has no frequencies.

    Example:
        rms_value = band_rms(wave, fs, 10.0, 100.0)
    """
    n     = sig.size
    freqs = np.fft.rfftfreq(n, d=1/fs)
    X     = np.fft.rfft(sig - sig.mean())
    P     = (np.abs(X)**2) / n
    mask  = (freqs >= f_lo) & (freqs <= f_hi)
    return np.sqrt(P[mask].sum()) if mask.any() else np.nan


def band_peak(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """
    Compute the peak absolute amplitude of a signal in a given frequency band.

    This function masks the FFT coefficients outside the specified band,
    performs an inverse FFT, and returns the maximum absolute amplitude.

    Args:
        sig (np.ndarray): Time-domain signal.
        fs (float): Sampling rate in Hz.
        f_lo (float): Lower bound of frequency band (Hz).
        f_hi (float): Upper bound of frequency band (Hz).

    Returns:
        float: Peak amplitude in the specified band.

    Example:
        peak_val = band_peak(wave, fs, 10.0, 1000.0)
    """
    n     = sig.size
    X     = np.fft.rfft(sig - sig.mean())
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mask  = (freqs >= f_lo) & (freqs <= f_hi)
    Xf    = np.zeros_like(X)
    Xf[mask] = X[mask]
    return np.abs(np.fft.irfft(Xf, n)).max()


def discover_records() -> pd.DataFrame:
    """
    Discover all .Single files, parse their timestamps, and initialize a DataFrame.

    This function scans RAR_DIR for `.Single` files, extracts the timestamp
    from each filename using a regex, and builds a DataFrame with columns for
    timestamp, filepath, DSP metrics, and status placeholders (all initialized
    to NA).

    Args:
        None

    Returns:
        pd.DataFrame: DataFrame sorted by timestamp with one row per file,
                      and columns for metrics and statuses.

    Example:
        df = discover_records()
    """
    records = []
    for fp in RAR_DIR.rglob("*.Single"):
        match = TIMESTAMP_PATTERN.search(fp.name)
        if not match:
            continue
        ts = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        rec = {"timestamp": ts, "filepath": str(fp.relative_to(PROJECT_ROOT))}
        for col in METRIC_COLS + STATUS_COLS:
            rec[col] = pd.NA
        records.append(rec)

    if not records:
        print(f"⚠ No .Single files found in {RAR_DIR}")
        exit(1)

    records.sort(key=lambda r: r["timestamp"])
    return pd.DataFrame.from_records(records)


def compute_and_plot(df: pd.DataFrame):
    """
    Compute DSP metrics for each record, plot the waveform, and save the plot.

    For each row in the DataFrame, this function reads the waveform, computes
    velocity RMS, crest factor, kurtosis, peak values, band-limited RMS and
    peak, then saves a time-domain plot to PLOT_DIR.

    Args:
        df (pd.DataFrame): DataFrame returned by discover_records(), to be
                           updated in-place with computed metrics.

    Returns:
        None

    Example:
        compute_and_plot(df)
    """
    for idx, row in df.iterrows():
        path = PROJECT_ROOT / row["filepath"]
        try:
            fs, wave = read_single_file(path)
        except Exception as e:
            print(f"⚠ Skipping {row['filepath']}: {e}")
            continue

        vrms = np.sqrt((wave**2).mean())
        peak = np.max(np.abs(wave))

        df.at[idx, "velocity_rms"]   = vrms
        df.at[idx, "crest_factor"]   = peak / vrms if vrms > 0 else np.nan
        df.at[idx, "kurtosis_opt"]   = stats.kurtosis(wave, fisher=False)
        df.at[idx, "peak_value_opt"] = peak

        df.at[idx, "rms_0_10hz"]     = band_rms(wave, fs,    0.1,    10)
        df.at[idx, "rms_10_100hz"]   = band_rms(wave, fs,   10,    100)
        df.at[idx, "rms_1_10khz"]    = band_rms(wave, fs, 1000,   10000)
        df.at[idx, "rms_10_25khz"]   = band_rms(wave, fs,10000,   25000)
        df.at[idx, "peak_10_1000hz"] = band_peak(wave, fs,   10,    1000)

        times = np.arange(len(wave)) / fs
        plt.figure(figsize=(8, 3))
        plt.plot(times, wave)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        title = Path(row["filepath"]).stem
        plt.title(title)
        out_png = PLOT_DIR / f"{title}.png"
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"🖼️ Saved plot: {out_png.relative_to(PROJECT_ROOT)}")


def save_metrics(df: pd.DataFrame):
    """
    Print the first 10 rows of metrics and save the full DataFrame to CSV.

    This function displays a preview of the computed metrics and writes the
    complete DataFrame to OUT_CSV.

    Args:
        df (pd.DataFrame): DataFrame with computed DSP metrics.

    Returns:
        None

    Example:
        save_metrics(df)
    """
    print("\n— First 10 rows of metrics —")
    print(df.head(10).to_string(index=False))
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Metrics saved to: {OUT_CSV.relative_to(PROJECT_ROOT)}")


def main():
    """
    Run the full computation pipeline.

    This function orchestrates directory setup, archive extraction,
    record discovery, metric computation, plotting, and saving of results.

    Args:
        None

    Returns:
        None

    Example:
        main()
    """
    ensure_directories()
    unpack_archives()
    df = discover_records()
    compute_and_plot(df)
    save_metrics(df)


if __name__ == "__main__":
    main()