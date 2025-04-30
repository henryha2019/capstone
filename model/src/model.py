import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_validate
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
    # Load and clean data
    df = pd.read_csv('Data/raw/8#Belt Conveyer_merged.csv')
    df = df.drop(columns=['Device'], errors='ignore').dropna()

    # Preprocessing: datetime → timestamp, location → one-hot
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime_ts'] = df['datetime'].view('int64') // 10**9
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

    # Define pipelines
    pipelines = {
        'Baseline': Pipeline([('pre', preprocessor),
                              ('reg', DummyRegressor(strategy='mean'))]),
        'Ridge': Pipeline([('pre', preprocessor),
                           ('reg', MultiOutputRegressor(Ridge(alpha=1.0)))]),
        'PolyRidge(deg=2)': Pipeline([('pre', preprocessor_poly2),
                                      ('reg', MultiOutputRegressor(Ridge(alpha=1.0)))]),
        'PolyRidge(deg=5)': Pipeline([('pre', preprocessor_poly5),
                                      ('reg', MultiOutputRegressor(Ridge(alpha=1.0)))]),
        'RandomForest': Pipeline([('pre', preprocessor),
                                  ('reg', MultiOutputRegressor(RandomForestRegressor(
                                      n_estimators=100, random_state=42)))]),
    }

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
        cv_res = cross_validate(pipe, X, y, cv=5, scoring=scoring, n_jobs=-1)
        summary = evaluate_cv(cv_res)
        summary.insert(0, 'Model', name)
        results_list.append(summary)

    cv_df = pd.concat(results_list, ignore_index=True)
    print("Cross-Validation Metrics (averaged):")
    print(cv_df.to_string(index=False))
    cv_df.to_csv('cv_metrics.csv', index=False)


if __name__ == '__main__':
    main()
