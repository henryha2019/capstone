import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense

class RNNRegressor(BaseEstimator, RegressorMixin):
    """Custom RNN model wrapper for scikit-learn compatibility."""
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


def evaluate_cv(cv_results):
    """
    Summarize cross-validation results into a DataFrame.
    Assumes negative MSE/MAE are returned, so we convert them back.
    """
    metrics = {
        'MSE': cv_results['test_neg_mean_squared_error'] * -1,
        'RMSE': np.sqrt(-cv_results['test_neg_mean_squared_error']),
        'MAE': cv_results['test_neg_mean_absolute_error'] * -1,
        'R2': cv_results['test_r2'],
        'Explained Variance': cv_results['test_explained_variance']
    }
    df = pd.DataFrame(metrics)
    return df.mean().to_frame().T


def main():
    # Load and clean data
    df = pd.read_csv('Data/process/8#Belt Conveyer_merged.csv')
    df = df.drop(columns=['Device'], errors='ignore').dropna()
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

    # Preprocessing pipeline
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Preprocess X
    X_processed = preprocessor.fit_transform(X)

    # Reshape for RNN: (samples, time_steps, features)
    X_rnn = X_processed.reshape(X_processed.shape[0], 1, X_processed.shape[1])

    # Define scoring
    scoring = {
        'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
        'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'explained_variance': make_scorer(explained_variance_score)
    }

    # Cross-validation setup
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize and evaluate RNN model
    rnn_model = RNNRegressor(
        input_shape=(X_rnn.shape[1], X_rnn.shape[2]),
        output_dim=y.shape[1],
        epochs=10,
        batch_size=32
    )

    cv_res = cross_validate(rnn_model, X_rnn, y, cv=tscv, scoring=scoring, n_jobs=-1)
    summary = evaluate_cv(cv_res)

    print("RNN Cross-Validation Metrics (averaged):")
    print(summary.to_string(index=False))
    summary.to_csv('rnn_cv_metrics.csv', index=False)


if __name__ == '__main__':
    main()
