import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, OneHotEncoder
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
            # Se per qualche ragione arrivasse ancora un valore sconosciuto
            # (ad esempio, perché non è stato sostituito da "Other" a monte),
            # qui potremmo decidere come gestirlo. Per semplicità:
            known_values = set(self.encoders[col].classes_)
            X[col] = X[col].apply(lambda x: x if x in known_values else "Other")
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
        
        # Converti in category
        df[col] = df[col].astype("category")

        # Forza la presenza di "Other" tra le categorie
        # (se non esiste già, la aggiunge senza errori)
        if "Other" not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories(["Other"])

        return df


    def fit(self, X, y=None):
        df = X.copy()
        df = self._clean(df)

        # Uniformiamo la logica: semplifichiamo sia le high_cardinality_cols
        # sia le labelencode_cols
        for col in self.high_cardinality_cols + self.labelencode_cols:
            if col in df.columns:
                df = self._simplify_categories(df, col)

        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.difference(['is_bot'])

        self.pipeline = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=True), 
             self.onehot_cols + self.high_cardinality_cols),
            ('labelencode', MultiColumnLabelEncoder(self.labelencode_cols), self.labelencode_cols),
            ('scale', MaxAbsScaler(), self.numerical_cols)
        ], remainder='passthrough')

        self.pipeline.fit(df, y)
        return self

    def transform(self, X):
        df = X.copy()
        df = self._clean(df)

        # Stessa logica di semplificazione anche in transform
        for col in self.high_cardinality_cols + self.labelencode_cols:
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
