import sys
import struct
import importlib.util
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred in moment calculation due to catastrophic cancellation"
)

import numpy as np
import pandas as pd
import pytest

# —————————— Load feature_eng_rar.py by path ——————————
SRC = Path(__file__).resolve().parent.parent / "archive" / "feature_eng_rar.py"
spec = importlib.util.spec_from_file_location("feature_eng_rar", str(SRC))
fer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fer)


def create_single_file(path: Path, fs: float, samples: np.ndarray):
    """Helper to write a .Single file with given fs and samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<f", fs))
        samples.astype("<f4").tofile(f)


def test_read_single_file(tmp_path):
    fs_in = 1234.5
    samples = np.array([0.0, 1.0, -2.0, 0.5], dtype=float)
    single = tmp_path / "device_20250101_000000.Single"
    create_single_file(single, fs_in, samples)

    fs_out, wave = fer.read_single_file(single)
    assert fs_out == pytest.approx(fs_in)
    np.testing.assert_allclose(wave, samples)


def test_band_rms_and_peak_zero_signal():
    sig = np.zeros(512)
    fs = 200.0
    # RMS of zero signal is 0
    assert fer.band_rms(sig, fs, 0.1, 50) == pytest.approx(0.0)
    # Peak of zero signal is 0
    assert fer.band_peak(sig, fs, 0.1, 50) == pytest.approx(0.0)


def test_band_peak_sine_wave():
    # create a 5Hz sine over 1s at 100Hz sampling
    fs = 100
    t = np.arange(0, 1, 1/fs)
    wave = np.sin(2 * np.pi * 5 * t)
    peak = fer.band_peak(wave, fs, 4.0, 6.0)
    assert peak == pytest.approx(1.0, abs=0.05)


def test_discover_and_compute(tmp_path, monkeypatch):
    # Set up fake project structure
    fake_root   = tmp_path / "project"
    fake_rar    = fake_root / "data" / "rar_files"
    fake_plots  = fake_root / "docs" / "images" / "rar_plots"
    fake_rar.mkdir(parents=True, exist_ok=True)
    fake_plots.mkdir(parents=True, exist_ok=True)

    # Two .Single files
    fs1, samples1 = 50.0, np.array([1.0, -1.0], float)
    fs2, samples2 = 100.0, np.array([1.0, 2.0, 3.0, 4.0])  # avoid constant signal
    f1 = fake_rar / "dev_20250102_121212.Single"
    f2 = fake_rar / "dev_20250103_111111.Single"
    create_single_file(f1, fs1, samples1)
    create_single_file(f2, fs2, samples2)

    # Patch module paths
    monkeypatch.setattr(fer, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(fer, "RAR_DIR", fake_rar)
    monkeypatch.setattr(fer, "PLOT_DIR", fake_plots)
    monkeypatch.setattr(fer, "OUT_CSV", fake_rar / "metrics_all.csv")

    # Run discovery
    df = fer.discover_records()
    assert len(df) == 2
    assert list(df["timestamp"]) == sorted(df["timestamp"])

    # Compute and plot
    fer.compute_and_plot(df)
    out1 = fake_plots / (f1.stem + ".png")
    out2 = fake_plots / (f2.stem + ".png")
    assert out1.exists() and out2.exists()

    # Check metric values for first file: RMS=1, crest=1
    assert df.at[0, "velocity_rms"] == pytest.approx(1.0)
    assert df.at[0, "crest_factor"] == pytest.approx(1.0)

    # Save and verify CSV
    fer.save_metrics(df)
    saved = pd.read_csv(fake_rar / "metrics_all.csv")
    for col in ["velocity_rms", "crest_factor", "kurtosis_opt", "peak_value_opt"]:
        assert col in saved.columns
