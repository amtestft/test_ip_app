import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ==== Custom LabelEncoder Wrapper for Pipelines ====
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = self.encoders[col].transform(X[col])
        return X

# ==== Custom Preprocessor ====
class ClickDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = None
        self.columns_to_drop = [
            'date', 'gclid', 'session_start', 'session_end', 'session_id', 'custom_client',
            'device_is_limited_ad_tracking', 'id',
            'session_duration', 'average_session_duration', 'total_page_views'
        ]
        self.onehot_cols = [
            'channel', 'utm_medium', 'utm_source', 'utm_term',
            'continent', 'device_category', 'device_os', 'browser'
        ]
        self.labelencode_cols = [
            'country', 'region', 'device_brand', 'device_os_version', 'device_language'
        ]
        self.targetencode_cols = [
            'city', 'device_model', 'browser_version'
        ]
        self.numerical_cols = None

    def fit(self, X, y=None):
        df = X.copy()
        df = self._clean(df)

        # Set numerical columns for scaling
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.difference(['is_bot'])

        # Build pipeline
        self.pipeline = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), self.onehot_cols),
            ('labelencode', MultiColumnLabelEncoder(self.labelencode_cols), self.labelencode_cols),
            ('targetencode', TargetEncoder(), self.targetencode_cols),
            ('scale', StandardScaler(), self.numerical_cols)
        ], remainder='passthrough')

        self.pipeline.fit(df, y)
        return self

    def transform(self, X):
        df = X.copy()
        df = self._clean(df)

        # Apply pipeline
        X_processed = self.pipeline.transform(df)
        # Get column names (optional, useful for converting back to DataFrame)
        return X_processed

    def _clean(self, df):
        # Drop columns with all missing values
        missing = df.isnull().sum() / len(df)
        df = df[df.columns[missing < 1]]

        # Drop irrelevant columns
        df = df.drop(columns=[col for col in self.columns_to_drop if col in df.columns], errors='ignore')

        # Create ID if session_id and custom_client exist
        if 'session_id' in df.columns and 'custom_client' in df.columns:
            df['id'] = df['session_id'].astype(str) + '_' + df['custom_client'].astype(str)

        # Ensure is_bot is binary int
        if 'is_bot' in df.columns:
            df['is_bot'] = df['is_bot'].astype(int)

        return df
