import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
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
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = self.encoders[col].transform(X[col].astype(str)).astype(int)
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
        self.high_cardinality_cols = [
            'city', 'device_model', 'browser_version'
        ]
        self.numerical_cols = None

    def _simplify_categories(self, df, col, threshold=0.01):
        df[col] = df[col].astype(str)
        freq = df[col].value_counts(normalize=True)
        common = freq[freq > threshold].index
        df[col] = df[col].apply(lambda x: x if x in common else 'Other')
        df[col] = df[col].astype("category")
        return df

    def fit(self, X, y=None):
        df = X.copy()
        df = self._clean(df)

        for col in self.high_cardinality_cols:
            if col in df.columns:
                df = self._simplify_categories(df, col)

        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.difference(['is_bot'])

        self.pipeline = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), self.onehot_cols + self.high_cardinality_cols),
            ('labelencode', MultiColumnLabelEncoder(self.labelencode_cols), self.labelencode_cols),
            ('scale', StandardScaler(), self.numerical_cols)
        ], remainder='passthrough')

        self.pipeline.fit(df, y)
        return self

    def transform(self, X):
        df = X.copy()
        df = self._clean(df)

        for col in self.high_cardinality_cols:
            if col in df.columns:
                df = self._simplify_categories(df, col)

        return self.pipeline.transform(df)

    def _clean(self, df):
        missing = df.isnull().sum() / len(df)
        df = df[df.columns[missing < 1]]
        df = df.drop(columns=[col for col in self.columns_to_drop if col in df.columns], errors='ignore')

        if 'session_id' in df.columns and 'custom_client' in df.columns:
            df['id'] = df['session_id'].astype(str) + '_' + df['custom_client'].astype(str)

        if 'is_bot' in df.columns:
            df['is_bot'] = df['is_bot'].astype(int)

        return df