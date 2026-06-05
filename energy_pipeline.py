import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class EnergyFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create consistent temporal features from Date, Start_Hour, and Source."""

    def fit(self, X, y=None):
        return self

    def _season_from_month(self, month):
        if month in [12, 1, 2]:
            return 'Winter'
        if month in [3, 4, 5]:
            return 'Spring'
        if month in [6, 7, 8]:
            return 'Summer'
        if month in [9, 10, 11]:
            return 'Fall'
        return np.nan

    def transform(self, X):
        X = X.copy()
        required_cols = ['Date', 'Start_Hour', 'Source']
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f'Missing required columns: {missing_cols}')

        date = pd.to_datetime(X['Date'], errors='coerce')
        start_hour = pd.to_numeric(X['Start_Hour'], errors='coerce')
        end_hour = (start_hour + 1) % 24
        duration = np.ones(len(X), dtype=float)

        out = pd.DataFrame(index=X.index)
        out['Source'] = X['Source'].astype(str)
        out['Start_Hour'] = start_hour.astype(float)
        out['End_Hour'] = end_hour.astype(float)
        out['duration'] = duration
        out['Day_of_Year'] = date.dt.dayofyear.astype(float)
        out['Day_Name'] = date.dt.day_name()
        out['Month'] = date.dt.month.astype(float)
        out['Month_Name'] = date.dt.month_name()
        out['Year'] = date.dt.year.astype(float)
        out['Date_Ordinal'] = (date - pd.Timestamp('1970-01-01')).dt.days.astype(float)
        out['Season'] = date.dt.month.map(self._season_from_month)

        out['hour_sin'] = np.sin(2 * np.pi * out['Start_Hour'] / 24)
        out['hour_cos'] = np.cos(2 * np.pi * out['Start_Hour'] / 24)
        out['dayyear_sin'] = np.sin(2 * np.pi * out['Day_of_Year'] / 366)
        out['dayyear_cos'] = np.cos(2 * np.pi * out['Day_of_Year'] / 366)
        out['month_sin'] = np.sin(2 * np.pi * out['Month'] / 12)
        out['month_cos'] = np.cos(2 * np.pi * out['Month'] / 12)
        out['is_weekend'] = out['Day_Name'].isin(['Saturday', 'Sunday']).astype(int)
        out['is_daytime'] = out['Start_Hour'].between(6, 18).astype(int)
        out['solar_daytime'] = ((out['Source'] == 'Solar') & (out['is_daytime'] == 1)).astype(int)
        out['solar_night'] = ((out['Source'] == 'Solar') & (out['is_daytime'] == 0)).astype(int)
        out['summer_solar_daytime'] = ((out['Season'] == 'Summer') & (out['Source'] == 'Solar') & (out['is_daytime'] == 1)).astype(int)
        return out

class EnergyPreprocessor(BaseEstimator, TransformerMixin):
    """OneHotEncoder plus numeric imputation in one reusable transformer."""

    def __init__(self, scale_numeric=False):
        self.scale_numeric = scale_numeric
        self.categorical_cols = ['Source', 'Day_Name', 'Month_Name', 'Season']
        self.numeric_cols = [
            'Start_Hour', 'End_Hour', 'duration', 'Day_of_Year', 'Month', 'Year', 'Date_Ordinal',
            'hour_sin', 'hour_cos', 'dayyear_sin', 'dayyear_cos', 'month_sin', 'month_cos',
            'is_weekend', 'is_daytime', 'solar_daytime', 'solar_night', 'summer_solar_daytime'
        ]

    def fit(self, X, y=None):
        try:
            self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.imputer_ = SimpleImputer(strategy='median')
        self.encoder_.fit(X[self.categorical_cols])
        numeric = self.imputer_.fit_transform(X[self.numeric_cols])
        if self.scale_numeric:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(numeric)
        else:
            self.scaler_ = None
        return self

    def transform(self, X):
        cat = self.encoder_.transform(X[self.categorical_cols])
        num = self.imputer_.transform(X[self.numeric_cols])
        if self.scaler_ is not None:
            num = self.scaler_.transform(num)
        return np.hstack([cat, num])

    def get_feature_names_out(self):
        cat_names = self.encoder_.get_feature_names_out(self.categorical_cols).tolist()
        return cat_names + self.numeric_cols
