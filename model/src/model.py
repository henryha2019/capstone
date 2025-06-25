"""
Feature engineering module: computes DSP metrics from waveform JSON files,
aggregates measurement and rating data into time buckets, and merges all
features into a final dataset for training and evaluation.
"""
import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import boto3
import io
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
    Summarize cross-validation results.

    Converts negative error metrics into positive and computes means.

    Args:
        cv_results: Output dict from sklearn.model_selection.cross_validate.

    Returns:
        DataFrame with one row of averaged metrics: MSE, RMSE, MAE, R2, Explained Variance.
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


def save_results(base_dir, device, best_models, aws_mode=False):
    """
    Persist best models, metrics, and prediction plots locally or to S3.

    Args:
        base_dir: Root directory for saving results locally.
        device: Device identifier for folder organization.
        best_models: Mapping target->(metrics_df, model_pipeline, (y_true, y_pred)).
        aws_mode: If True, upload outputs to S3 bucket 'brilliant-automation-capstone'.
    """
    s3_bucket = 'brilliant-automation-capstone'
    
    if aws_mode:
        s3 = boto3.client('s3')
        models_prefix = f"results/models/{device}/"
        metrics_prefix = f"results/metrics/{device}/"
        plots_prefix = f"results/plots/{device}/"
    else:
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
        if aws_mode:
            model_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            model_buffer.seek(0)
            s3_key = f"{models_prefix}{target}_best_model.pkl"
            s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=model_buffer.getvalue())
            print(f"Saved best model for '{target}' to s3://{s3_bucket}/{s3_key}")
        else:
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
        
        if aws_mode:
            plot_buffer = io.BytesIO()
            plt.savefig(plot_buffer, format='png')
            plot_buffer.seek(0)
            s3_key = f"{plots_prefix}{target}_predictions.png"
            s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=plot_buffer.getvalue())
            print(f"Saved prediction plot for '{target}' to s3://{s3_bucket}/{s3_key}")
        else:
            plot_path = plots_dir / f"{target}_predictions.png"
            plt.savefig(plot_path)
            print(f"Saved prediction plot for '{target}' to {plot_path}")
        
        plt.close()

    # Save all metrics to a single CSV file
    combined_metrics = pd.concat(metrics_list)
    
    if aws_mode:
        csv_buffer = io.StringIO()
        combined_metrics.to_csv(csv_buffer, index=False)
        s3_key = f"{metrics_prefix}cv_metrics.csv"
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())
        print(f"Saved cross-validation metrics to s3://{s3_bucket}/{s3_key}")
    else:
        metrics_file = metrics_dir / "cv_metrics.csv"
        combined_metrics.to_csv(metrics_file, index=False)
        print(f"Saved cross-validation metrics to {metrics_file}")


def main():
    """
    Train and evaluate a suite of regression models on processed features.

    Steps:
      1. Parse arguments for model selection, tuning, device, and aws flag.
      2. Load feature dataset from local or S3.
      3. Preprocess features and targets.
      4. Optionally perform hyperparameter search.
      5. Evaluate via time-series cross-validation.
      6. Save the best models and metrics.
    """
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
    parser.add_argument(
        "--aws",
        action="store_true",
        help="Read data from S3 bucket and save results to S3 instead of local directories"
    )
    args = parser.parse_args()

    result_dir = Path(__file__).resolve().parent.parent
    s3_bucket = 'brilliant-automation-capstone'

    # Load dataset from local or S3
    if args.aws:
        print(f"Loading data from S3 bucket: {s3_bucket}")
        s3 = boto3.client('s3')
        s3_key = f"processed/{args.device}_full_features.csv"
        try:
            obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), parse_dates=['datetime'])
            print(f"Loaded data from s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"Failed to load data from S3: {e}")
            return
    else:
        # Load dataset (input data stays in the data directory)
        data_path = Path('data') / 'processed' / f"{args.device}_full_features.csv"
        if not data_path.exists():
            print(f"File not found: {data_path}")
            return
        df = pd.read_csv(data_path, parse_dates=['datetime'])
        print(f"Loaded data from {data_path}")

    # Debug: Print column info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Drop unused columns and handle missing data
    columns_to_drop = ['filepath', 'sensor_id', 'bucket_id', 'bucket_end', 's3_key']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    print(f"Dropping columns: {existing_columns_to_drop}")
    
    df = df.drop(columns=existing_columns_to_drop).dropna()
    print(f"Shape after dropping columns and NaN: {df.shape}")

    # Optional: Subsample large datasets
    if len(df) > 100_000:
        print("Subsampling dataset to reduce memory load...")
        df = df.sample(frac=0.5, random_state=42)

    # Feature engineering
    df['datetime_ts'] = df['datetime'].astype('int64') // 10 ** 9
    df = pd.get_dummies(df, columns=['location', 'wave_code'], prefix=['loc', 'wave'])

    # Separate features (X) and target values (y)
    rating_cols = [col for col in df.columns if col.endswith('_rating')]
    print(f"Rating columns found: {rating_cols}")
    
    X = df.drop(columns=rating_cols + ['datetime'])
    y = df[rating_cols]
    
    # Debug: Print feature info
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Check for any non-numeric columns in X
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"WARNING: Non-numeric columns found in features: {non_numeric_cols}")
        print("Sample values from non-numeric columns:")
        for col in non_numeric_cols[:3]:  # Show first 3 non-numeric columns
            print(f"  {col}: {X[col].iloc[:5].tolist()}")
        
        # Try to drop problematic string columns
        print("Attempting to drop string columns...")
        X = X.select_dtypes(include=[np.number])
        print(f"Features shape after dropping non-numeric columns: {X.shape}")
    
    print(f"Final feature columns: {list(X.columns)}")

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
    save_results(result_dir, args.device, best_models, aws_mode=args.aws)


if __name__ == "__main__":
    main()