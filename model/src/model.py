import pandas as pd
import numpy as np
import argparse
import boto3
import io
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


def evaluate_cv(cv_results):
    """
    Summarize cross-validation results into a DataFrame.
    Assumes negative MSE/MAE are returned, so we convert them back.
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
    parser = argparse.ArgumentParser(description="Train and evaluate machine learning models for device data.")
    parser.add_argument("--model", choices=['Baseline', 'Ridge', 'PolyRidgeDegree2', 'PolyRidgeDegree5', 'RandomForest', 'all'], 
                      default='all', help="Model to train (default: all)")
    parser.add_argument("--aws", action="store_true", help="Read/write data from/to S3 bucket")
    parser.add_argument("--device", default="8#Belt Conveyer", help="Device name (default: 8#Belt Conveyer)")
    args = parser.parse_args()

    # Load and clean data
    if args.aws:
        s3 = boto3.client('s3')
        bucket = 'brilliant-automation-capstone'
        key = f"process/{args.device}_merged.csv"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        except Exception as e:
            print(f"Error reading from S3: {e}")
            return
    else:
        try:
            df = pd.read_csv(f'Data/process/{args.device}_merged.csv')
        except FileNotFoundError:
            print(f"Error: Could not find data file for device {args.device}")
            return

    df = df.drop(columns=['Device'], errors='ignore').dropna()
    # Using a sample dataset comment line below for full data
    df = df[:100000]

    # Preprocessing: datetime → timestamp, location → one-hot
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime_ts'] = df['datetime'].astype('int64') // 10**9
    df = pd.get_dummies(df, columns=['location'], prefix='loc')

    # Features and targets
    numeric_features = [
        'datetime_ts',
        'High-Frequency Acceleration',
        'Low-Frequency Acceleration Z',
        'Temperature',
        'Vibration Velocity Z'
    ]
    categorical_features = [c for c in df.columns if c.startswith('loc_')]
    X = df[numeric_features + categorical_features]
    y = df.drop(columns=numeric_features + categorical_features + ['datetime'])

    # Base preprocessors
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])

    # Polynomial preprocessors
    numeric_poly2 = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])
    numeric_poly5 = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=5, include_bias=False))
    ])

    # ColumnTransformers for each
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    preprocessor_poly2 = ColumnTransformer([
        ('num', numeric_poly2, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    preprocessor_poly5 = ColumnTransformer([
        ('num', numeric_poly5, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Define all possible pipelines
    all_pipelines = {
        'Baseline': Pipeline([('pre', preprocessor),
                            ('reg', DummyRegressor(strategy='mean'))]),
        'Ridge': Pipeline([('pre', preprocessor),
                         ('reg', MultiOutputRegressor(Ridge(alpha=1.0)))]),
        'PolyRidgeDegree2': Pipeline([('pre', preprocessor_poly2),
                         ('reg', MultiOutputRegressor(Ridge(alpha=1.0)))]),
        'PolyRidgeDegree5': Pipeline([('pre', preprocessor_poly5),
                         ('reg', MultiOutputRegressor(Ridge(alpha=1.0)))]),
        'RandomForest': Pipeline([('pre', preprocessor),
                       ('reg', MultiOutputRegressor(RandomForestRegressor(
                           n_estimators=100, random_state=42)))])
    }

    # Select pipelines based on model argument
    if args.model == 'all':
        pipelines = all_pipelines
    else:
        pipelines = {args.model: all_pipelines[args.model]}

    # Cross-validation setup
    scoring = {
        'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
        'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'explained_variance': make_scorer(explained_variance_score)
    }

    # Run CV for each model
    results_list = []
    for name, pipe in pipelines.items():
        tscv = TimeSeriesSplit(n_splits=5)
        cv_res = cross_validate(pipe, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
        summary = evaluate_cv(cv_res)
        summary.insert(0, 'Model', name)
        results_list.append(summary)

    cv_df = pd.concat(results_list, ignore_index=True)
    print("Cross-Validation Metrics (averaged):")
    print(cv_df.to_string(index=False))

    # Save results
    if args.aws:
        s3 = boto3.client('s3')
        bucket = 'brilliant-automation-capstone'
        key = f"results/{args.device}_{args.model}_cv_metrics.csv"
        csv_buffer = io.StringIO()
        cv_df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        print(f"Results saved to s3://{bucket}/{key}")
    else:
        output_file = f'{args.device}_{args.model}_cv_metrics.csv'
        cv_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()