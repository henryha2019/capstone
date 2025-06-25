import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for plotting

import pytest
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

import model  # assumes your code is in model.py at project root

# import the functions under test
from model import evaluate_cv, save_results


def test_evaluate_cv():
    # build a fake cross-val result
    cv_results = {
        'test_neg_mean_squared_error': np.array([-4.0, -9.0]),
        'test_neg_mean_absolute_error': np.array([-2.0, -3.0]),
        'test_r2': np.array([0.8, 0.6]),
        'test_explained_variance': np.array([0.7, 0.5]),
    }

    df = evaluate_cv(cv_results)

    # should be a single‐row DataFrame
    assert df.shape == (1, 5)
    row = df.iloc[0]

    # MSE: mean of [4,9] = 6.5
    assert pytest.approx(row['MSE'], rel=1e-6) == 6.5
    # RMSE: mean of [2,3] = 2.5
    assert pytest.approx(row['RMSE'], rel=1e-6) == 2.5
    # MAE: mean of [2,3] = 2.5
    assert pytest.approx(row['MAE'], rel=1e-6) == 2.5
    # R2: mean of [0.8,0.6] = 0.7
    assert pytest.approx(row['R2'], rel=1e-6) == 0.7
    # Explained Variance: mean of [0.7,0.5] = 0.6
    assert pytest.approx(row['Explained Variance'], rel=1e-6) == 0.6


def make_dummy_best_models(tmp_path):
    """
    Create a minimal best_models dict for save_results:
      target -> (metrics_df, pipeline, (y_true, y_pred))
    """
    # dummy metrics DataFrame with at least one column
    metrics = pd.DataFrame({'some_metric': [1.23]})
    # a simple sklearn Pipeline with a named_steps['model']
    pipeline = Pipeline([('model', DummyRegressor())])

    # create a dummy prediction pair
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.1, 0.9, 2.1])

    return {'dummy_target': (metrics, pipeline, (y_true, y_pred))}


def test_save_results(tmp_path, monkeypatch):
    # prepare a fake best_models
    best_models = make_dummy_best_models(tmp_path)

    # call save_results in local (aws_mode=False)
    save_results(str(tmp_path), device='DEV1', best_models=best_models, aws_mode=False)

    # check that model .pkl, plot .png and metrics CSV were created
    base = Path(tmp_path) / "results"

    model_file = base / "models" / "DEV1" / "dummy_target_best_model.pkl"
    plot_file  = base / "plots"  / "DEV1" / "dummy_target_predictions.png"
    metrics_csv = base / "metrics" / "DEV1" / "cv_metrics.csv"

    assert model_file.exists(),  f"Missing model file at {model_file}"
    assert plot_file.exists(),   f"Missing plot file at {plot_file}"
    assert metrics_csv.exists(), f"Missing metrics CSV at {metrics_csv}"

    # sanity‐check CSV contents
    df = pd.read_csv(metrics_csv)
    # should have at least the columns inserted by save_results
    assert 'Best Model' in df.columns
    assert 'some_metric' in df.columns
    assert df.iloc[0]['Best Model'] == 'DummyRegressor'
