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
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


def load_and_clean_data(device: str, aws: bool) -> pd.DataFrame:
    """
    Load and clean the dataset for a given device.

    Reads a CSV from S3 or local filesystem, drops the 'Device' column, and removes rows with missing values.

    Args:
        device (str): Name of the device, used to construct the filename.
        aws (bool): If True, reads from S3 bucket 'brilliant-automation-capstone'; else reads locally.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for preprocessing.
    """
    if aws:
        s3 = boto3.client('s3')
        bucket = 'brilliant-automation-capstone'
        key = f"process/{device}_merged.csv"
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    else:
        df = pd.read_csv(f'data/processed/{device}_merged.csv')
    df = df.drop(columns=['Device'], errors='ignore').dropna()
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw DataFrame by encoding datetime and location.

    Converts 'datetime' to UNIX timestamp and one-hot encodes 'location'.

    Args:
        df (pd.DataFrame): Raw DataFrame containing 'datetime' and 'location' columns.

    Returns:
        pd.DataFrame: Transformed DataFrame with new features.
    """
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime_ts'] = df['datetime'].astype('int64') // 10**9
    df = pd.get_dummies(df, columns=['location'], prefix='loc')
    return df


def get_feature_target(df: pd.DataFrame):
    """
    Split DataFrame into features (X) and target matrix (y).

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        tuple: (X, y, numeric_features, categorical_features)
    """
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
    return X, y, numeric_features, categorical_features


def evaluate_cv(cv_results: dict) -> pd.DataFrame:
    """
    Summarize cross-validation results into averaged metrics.

    Args:
        cv_results (dict): Output from cross_validate containing negative errors and metrics.

    Returns:
        pd.DataFrame: Single-row DataFrame with averaged MSE, RMSE, MAE, R2, and Explained Variance.
    """
    mse = -cv_results['test_neg_mean_squared_error']
    mae = -cv_results['test_neg_mean_absolute_error']
    return pd.DataFrame({
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'R2': cv_results['test_r2'],
        'Explained Variance': cv_results['test_explained_variance']
    }).mean().to_frame().T


def save_results(cv_df: pd.DataFrame, device: str, model_name: str, aws: bool) -> None:
    """
    Save cross-validation results to CSV locally or on S3.

    Args:
        cv_df (pd.DataFrame): DataFrame of CV metrics.
        device (str): Device identifier used in filename.
        model_name (str): Model identifier used in filename.
        aws (bool): If True, uploads to S3 under 'results/'.
    """
    filename = f"{device}_{model_name}_cv_metrics.csv"
    if aws:
        csv_buffer = io.StringIO()
        cv_df.to_csv(csv_buffer, index=False)
        s3 = boto3.client('s3')
        s3.put_object(Bucket='brilliant-automation-capstone', Key=f"results/{filename}", Body=csv_buffer.getvalue())
        print(f"Results saved to s3://brilliant-automation-capstone/results/{filename}")
    else:
        output_dir = 'model/archive/result'
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        cv_df.to_csv(path, index=False)
        print(f"Results saved to {path}")


def main():
    """
    Orchestrate data loading, preprocessing, model evaluation, and result saving.

    Parses command-line arguments, runs CV for selected models, and outputs metrics.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate ML models for device data.")
    parser.add_argument("--model", choices=['Baseline','Ridge','PolyRidgeDegree2','PolyRidgeDegree5','RandomForest','all'],
                        default='all', help="Model to train (default: all)")
    parser.add_argument("--aws", action="store_true", help="Read/write data from/to S3")
    parser.add_argument("--device", default="8#Belt Conveyer", help="Device name")
    args = parser.parse_args()

    df = load_and_clean_data(args.device, args.aws)
    df = df.iloc[:10000]  # <-- limit to first 10k rows
    df = preprocess(df)
    X, y, num_feats, cat_feats = get_feature_target(df)

    # Define preprocessors
    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    poly2_pipe = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(2, include_bias=False))])
    poly5_pipe = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(5, include_bias=False))])
    preprocessors = {
        'Baseline': ColumnTransformer([('num', num_pipe, num_feats), ('cat', cat_pipe, cat_feats)]),
        'Ridge': ColumnTransformer([('num', num_pipe, num_feats), ('cat', cat_pipe, cat_feats)]),
        'PolyRidgeDegree2': ColumnTransformer([('num', poly2_pipe, num_feats), ('cat', cat_pipe, cat_feats)]),
        'PolyRidgeDegree5': ColumnTransformer([('num', poly5_pipe, num_feats), ('cat', cat_pipe, cat_feats)]),
        'RandomForest': ColumnTransformer([('num', num_pipe, num_feats), ('cat', cat_pipe, cat_feats)])
    }

    models = {
        'Baseline': DummyRegressor(strategy='mean'),
        'Ridge': Ridge(alpha=1.0),
        'PolyRidgeDegree2': Ridge(alpha=1.0),
        'PolyRidgeDegree5': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    selected = models if args.model=='all' else {args.model: models[args.model]}

    scoring = {
        'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
        'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'explained_variance': make_scorer(explained_variance_score)
    }

    results = []
    for name, mdl in selected.items():
        for target in y.columns:
            pipe = Pipeline([('pre', preprocessors[name]), ('reg', mdl)])
            cv_res = cross_validate(pipe, X, y[target], cv=TimeSeriesSplit(n_splits=5), scoring=scoring, n_jobs=-1)
            summary = evaluate_cv(cv_res)
            summary.insert(0, 'Target', target)
            summary.insert(0, 'Model', name)
            results.append(summary)

    cv_df = pd.concat(results, ignore_index=True)
    print(cv_df.to_string(index=False))
    save_results(cv_df, args.device, args.model, args.aws)


if __name__ == '__main__':
    main()
