#!/usr/bin/env python3
# model.py

import os
import io
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import boto3
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


def evaluate_cv(cv_results):
    """
    Summarize cross-validation results into a DataFrame.
    Assumes negative MSE/MAE in cv_results, so we convert them back.
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


def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate ML models on joined DSP+rating features"
    )
    parser.add_argument(
        "--model",
        choices=['Baseline', 'Ridge', 'PolyRidgeDegree2', 'RandomForest', 'XGBoost', 'SVR', 'RuleTree', 'all'],
        default='all',
        help="Which model to train (default: all)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform randomized hyperparameter tuning"
    )
    parser.add_argument(
        "--aws",
        action="store_true",
        help="Read/write data from S3 instead of local disk"
    )
    parser.add_argument(
        "--device",
        default="8#Belt Conveyer",
        help="Device name (affects file paths/key names)"
    )
    args = parser.parse_args()

    # Setup output directory
    result_dir = Path('Data') / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load full_features.csv
    if args.aws:
        bucket = 'brilliant-automation-capstone'
        key    = f"process/{args.device}_full_features.csv"
        try:
            obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)
            df  = pd.read_csv(io.BytesIO(obj['Body'].read()), parse_dates=['datetime'])
        except Exception as e:
            print("S3 read error:", e)
            return
    else:
        path = Path('Data') / 'process' / f"{args.device}_full_features.csv"
        if not path.exists():
            print("File not found:", path)
            return
        df = pd.read_csv(path, parse_dates=['datetime'])

    # Drop unused columns and missing data
    df = df.drop(columns=['filepath', 'sensor_id', 'bucket_id', 'bucket_end'], errors='ignore').dropna()

    # Feature engineering
    df['datetime_ts'] = df['datetime'].astype('int64') // 10**9
    df = pd.get_dummies(df, columns=['location', 'wave_code'], prefix=['loc', 'wave'])

    # Prepare X and multi-output y
    rating_cols = [c for c in df.columns if c.endswith('_rating')]
    X = df.drop(columns=rating_cols + ['datetime'])
    y = df[rating_cols]

    # Identify numeric vs categorical features
    cat_prefixes = ('loc_', 'wave_')
    categorical_features = [c for c in X.columns if c.startswith(cat_prefixes)]
    numeric_features     = [c for c in X.columns if c not in categorical_features]

    # Define prototype preprocessors
    scaler = StandardScaler()
    poly2  = PolynomialFeatures(degree=2, include_bias=False)
    preprocessors = {
        'Baseline': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'Ridge': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'PolyRidgeDegree2': ColumnTransformer([('num', Pipeline([('scale', scaler), ('poly', poly2)]), numeric_features)], remainder='passthrough'),
        'RandomForest': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'XGBoost': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'SVR': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough'),
        'RuleTree': ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough')
    }

    # Define prototype models
    models = {
        'Baseline': DummyRegressor(strategy='mean'),
        'Ridge': Ridge(),
        'PolyRidgeDegree2': Ridge(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror'),
        'SVR': SVR(),
        'RuleTree': DecisionTreeRegressor(max_depth=1, random_state=42)
    }

    # Hyperparameter distributions for tuning
    param_distributions = {
        'Ridge': {'reg__alpha': loguniform(1e-3, 1e3)},
        'PolyRidgeDegree2': {'reg__alpha': loguniform(1e-3, 1e3)},
        'RandomForest': {'reg__n_estimators': randint(50, 200), 'reg__max_depth': randint(3, 20)},
        'XGBoost': {'reg__n_estimators': randint(50, 200), 'reg__max_depth': randint(3, 20), 'reg__learning_rate': loguniform(1e-3, 1e-1)},
        'SVR': {'reg__C': loguniform(1e-2, 1e2), 'reg__gamma': ['scale', 'auto']},
        'RuleTree': {'reg__max_depth': randint(1, 5)}
    }

    # Select which models to run
    selected = models if args.model == 'all' else {args.model: models[args.model]}

    # Define scoring metrics
    scoring = {
        'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
        'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'explained_variance': make_scorer(explained_variance_score)
    }

    # Cross-validate and optionally tune
    results = []
    tscv    = TimeSeriesSplit(n_splits=5)
    best_estimators = {}

    for name, prototype in selected.items():
        for target in y.columns:
            # Clone fresh preprocessor and model
            pre = clone(preprocessors[name])
            mdl = clone(models[name])
            pipe = Pipeline([('pre', pre), ('reg', mdl)])

            # Hyperparameter tuning if requested
            if args.tune and name in param_distributions:
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions[name],
                    n_iter=20,
                    cv=tscv,
                    scoring='r2',
                    n_jobs=-1
                )
                search.fit(X, y[target])
                best_model = search.best_estimator_
            else:
                best_model = pipe.fit(X, y[target])

            best_estimators[(name, target)] = best_model

            # Evaluate with CV
            cv_res = cross_validate(best_model, X, y[target], cv=tscv, scoring=scoring, n_jobs=-1)
            summary = evaluate_cv(cv_res)
            summary.insert(0, 'Target', target)
            summary.insert(0, 'Model', name)
            results.append(summary)

    # Aggregate and display CV results
    cv_df = pd.concat(results, ignore_index=True)
    print("\nCross-Validation Metrics (averaged):")
    print(cv_df.to_string(index=False))

    # Save CV results locally to Data/result
    metrics_fn = result_dir / f"{args.device}_{args.model}_cv_metrics.csv"
    cv_df.to_csv(metrics_fn, index=False)
    print(f"Saved CV results to {metrics_fn}")

    # Save to S3 if requested
    if args.aws:
        bucket = 'brilliant-automation-capstone'
        out_key = f"results/{args.device}_{args.model}_cv_metrics.csv"
        buf = io.StringIO()
        cv_df.to_csv(buf, index=False)
        boto3.client('s3').put_object(Bucket=bucket, Key=out_key, Body=buf.getvalue())
        print(f"Also saved CV results to s3://{bucket}/{out_key}")

    # Determine the best model per target by max R² (excluding Baseline)
    no_baseline = cv_df[cv_df['Model'] != 'Baseline']
    best_per_target = (
        no_baseline.loc[no_baseline.groupby('Target')['R2'].idxmax()]
             .set_index('Target')['Model']
             .to_dict()
    )

    # Persist only the winners with plots to Data/result
    for (name, target), model in best_estimators.items():
        if best_per_target.get(target) != name:
            continue
        # Save the model
        model_fn = result_dir / f"{args.device}_{name}_{target}_model.pkl"
        joblib.dump(model, model_fn)
        print(f"Saved best model for {target}: {model_fn}")

        # Plot actual vs. predicted
        y_true = y[target].values
        y_pred = model.predict(X)
        plt.figure()
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f"Best: {name} - {target}")
        plt.xlabel('Sample index')
        plt.ylabel(target)
        plt.legend()
        plot_fn = result_dir / f"{args.device}_{name}_{target}_plot.png"
        plt.savefig(plot_fn)
        plt.close()
        print(f"Saved plot for {target}: {plot_fn}")

if __name__ == "__main__":
    main()