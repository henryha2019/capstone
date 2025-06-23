#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from scipy.stats import loguniform, randint
from joblib import parallel_backend


def evaluate_cv(cv_results):
    """
    Summarizes cross-validation results into a DataFrame.
    Converts negative metrics to their positive equivalents.
    """
    metrics = {
        'MSE': -cv_results['test_neg_mean_squared_error'],
        'RMSE': np.sqrt(-cv_results['test_neg_mean_squared_error']),
        'MAE': -cv_results['test_neg_mean_absolute_error'],
        'R2': cv_results['test_r2'],
        'Explained Variance': cv_results['test_explained_variance']
    }
    df = pd.DataFrame(metrics)
    return df.mean().to_frame().T


def save_results(base_dir, device, best_models):
    """
    Saves the best models, cross-validation metrics, and prediction plots for each target feature.
    Stores results in 'model/results'.
    """
    # Define the results directory relative to the base directory
    results_dir = Path(base_dir) / "results"
    models_dir = results_dir / "models" / device
    metrics_dir = results_dir / "metrics" / device
    plots_dir = results_dir / "plots" / device

    # Create directories if they do not exist
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics with the associated best model name
    metrics_list = []
    for target, (metrics, model, (y_true, y_pred)) in best_models.items():
        metrics = metrics.copy()  # Ensure we don't overwrite the original metrics
        metrics.insert(0, 'Best Model', type(model.named_steps['model']).__name__)  # Add best model name
        metrics_list.append(metrics)

        # Save the model
        model_path = models_dir / f"{target}_best_model.pkl"
        joblib.dump(model, model_path)
        print(f"Saved best model for '{target}' to {model_path}")

        # Generate and save prediction plot
        plt.figure()
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f"Best Model - {target}")
        plt.xlabel('Sample Index')
        plt.ylabel(target)
        plt.legend()
        plot_path = plots_dir / f"{target}_predictions.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved prediction plot for '{target}' to {plot_path}")

    # Save all metrics to a single CSV file
    metrics_file = metrics_dir / "cv_metrics.csv"
    combined_metrics = pd.concat(metrics_list)
    combined_metrics.to_csv(metrics_file, index=False)
    print(f"Saved cross-validation metrics to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate ML models on device data.")
    parser.add_argument(
        "--model",
        choices=['Baseline', 'Ridge', 'PolyRidgeDegree2', 'RandomForest', 'XGBoost', 'SVR', 'RuleTree', 'all'],
        default='all',
        help="Specify which model to train (default: all)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform randomized hyperparameter tuning"
    )
    parser.add_argument(
        "--device",
        default="8_Belt_Conveyer",
        help="Device name (used for file paths and output organization)"
    )
    args = parser.parse_args()

    result_dir = Path(__file__).resolve().parent.parent

    # Load dataset (input data stays in the data directory)
    data_path = Path('data') / 'processed' / f"{args.device}_full_features.csv"
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return
    df = pd.read_csv(data_path, parse_dates=['datetime'])

    # Drop unused columns and handle missing data
    df = df.drop(columns=['filepath', 'sensor_id', 'bucket_id', 'bucket_end'], errors='ignore').dropna()

    # Optional: Subsample large datasets
    if len(df) > 100_000:
        print("Subsampling dataset to reduce memory load...")
        df = df.sample(frac=0.5, random_state=42)

    # Feature engineering
    df['datetime_ts'] = df['datetime'].astype('int64') // 10 ** 9
    df = pd.get_dummies(df, columns=['location', 'wave_code'], prefix=['loc', 'wave'])

    # Separate features (X) and target values (y)
    rating_cols = [col for col in df.columns if col.endswith('_rating')]
    X = df.drop(columns=rating_cols + ['datetime'])
    y = df[rating_cols]

    # Dynamically determine parallelism
    num_cores = os.cpu_count()
    n_jobs_tuned = max(1, num_cores // 2)

    # Cross-validation setup
    tscv = TimeSeriesSplit(n_splits=3)  # Reduce splits to reduce processing load
    scoring = {
        'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
        'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'explained_variance': make_scorer(explained_variance_score)
    }

    # Preprocessing pipelines and models
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    poly2 = PolynomialFeatures(degree=2, include_bias=False)

    preprocessors = {
        'Baseline': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'Ridge': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'PolyRidgeDegree2': ColumnTransformer(
            [('num', Pipeline([('scale', scaler), ('poly', poly2)]), numeric_features)], remainder='passthrough'),
        'RandomForest': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'XGBoost': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'SVR': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'RuleTree': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough')
    }

    models = {
        'Baseline': DummyRegressor(strategy='mean'),
        'Ridge': Ridge(),
        'PolyRidgeDegree2': Ridge(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=1),
        'SVR': SVR(),
        'RuleTree': DecisionTreeRegressor(max_depth=1, random_state=42)
    }

    # Hyperparameter tuning ranges
    param_distributions = {
        'Ridge': {'model__alpha': loguniform(1e-3, 1e3)},
        'PolyRidgeDegree2': {'model__alpha': loguniform(1e-3, 1e3)},
        'RandomForest': {'model__n_estimators': randint(50, 200), 'model__max_depth': randint(3, 20)},
        'XGBoost': {'model__n_estimators': randint(50, 150), 'model__max_depth': [3, 4, 5],
                    'model__learning_rate': loguniform(0.01, 0.1)},
        'SVR': {'model__C': loguniform(1e-2, 1e2), 'model__gamma': ['scale', 'auto']},
        'RuleTree': {'model__max_depth': randint(1, 3)}
    }

    # Model training and evaluation
    best_models = {}
    for target in y.columns:
        best_model = None
        best_score = -np.inf
        best_metrics = None
        y_true, y_pred = None, None

        for name, model in models.items():
            pipeline = Pipeline([('preprocessor', clone(preprocessors[name])), ('model', clone(model))])

            with parallel_backend('threading', n_jobs=n_jobs_tuned):  # Thread-safe parallelism
                if args.tune and name in param_distributions:
                    search = RandomizedSearchCV(
                        pipeline,
                        param_distributions[name],
                        n_iter=10,
                        cv=tscv,
                        scoring='r2',
                        n_jobs=n_jobs_tuned
                    )
                    search.fit(X, y[target])
                    pipeline = search.best_estimator_

                pipeline.fit(X, y[target])
                cv_results = cross_validate(pipeline, X, y[target], cv=tscv, scoring=scoring, n_jobs=n_jobs_tuned)

            r2_mean = np.mean(cv_results['test_r2'])

            if r2_mean > best_score:
                best_score = r2_mean
                best_model = pipeline
                best_metrics = evaluate_cv(cv_results)
                y_true = y[target].values
                y_pred = pipeline.predict(X)

        # Store the best model and results
        best_metrics.insert(0, 'Target', target)
        best_models[target] = (best_metrics, best_model, (y_true, y_pred))

    # Save the results
    save_results(result_dir, args.device, best_models)


if __name__ == "__main__":
    main()