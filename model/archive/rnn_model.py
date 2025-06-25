import pandas as pd
import numpy as np
import argparse
import boto3
import io
import os
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense


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
    return pd.get_dummies(df, columns=['location'], prefix='loc')


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


class RNNRegressor(BaseEstimator, RegressorMixin):
    """
    Custom RNN model wrapper for scikit-learn compatibility.

    Args:
        input_shape (tuple): Shape of input sequence (timesteps, features).
        output_dim (int): Number of output targets.
        epochs (int): Training epochs.
        batch_size (int): Training batch size.
    """
    def __init__(self, input_shape, output_dim, epochs=10, batch_size=32):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(SimpleRNN(50, activation='relu'))
        model.add(Dense(self.output_dim))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit(self, X, y):
        """Fit the RNN model."""
        self.model = self.build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        """Make predictions with the fitted RNN model."""
        return self.model.predict(X)


def evaluate_cv_results(cv_results: dict) -> pd.DataFrame:
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


def save_results(cv_df: pd.DataFrame, filename: str) -> None:
    """
    Save cross-validation results to CSV.

    Args:
        cv_df (pd.DataFrame): DataFrame of CV metrics.
        filename (str): Output CSV filename.
    """
    cv_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main():
    """
    Run cross-validation on RNNRegressor.
    """
    df = pd.read_csv('data/processed/8#Belt Conveyer_merged.csv')
    df = df.drop(columns=['Device'], errors='ignore').dropna()
    df = df[:100000]

    df = preprocess(df)
    X, y, num_feats, cat_feats = get_feature_target(df)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])
    X_proc = preprocessor.fit_transform(X)
    X_rnn = X_proc.reshape(X_proc.shape[0], 1, X_proc.shape[1])

    scoring = {
        'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
        'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'explained_variance': make_scorer(explained_variance_score)
    }
    tscv = TimeSeriesSplit(n_splits=5)

    rnn = RNNRegressor(
        input_shape=(X_rnn.shape[1], X_rnn.shape[2]),
        output_dim=y.shape[1],
        epochs=10,
        batch_size=32
    )

    cv_res = cross_validate(rnn, X_rnn, y, cv=tscv, scoring=scoring, n_jobs=-1)
    summary = evaluate_cv_results(cv_res)

    print("RNN Cross-Validation Metrics (averaged):")
    print(summary.to_string(index=False))
    save_results(summary, 'model/archive/result/rnn_cv_metrics.csv')


if __name__ == '__main__':
    main()
