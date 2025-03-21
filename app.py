import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import plotly.express as px
import base64
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from processor import MultiColumnLabelEncoder, ClickDataPreprocessor

# -------------------------------
# Custom transformer for feature selection (must match train_model.py)
# -------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, indices):
        self.indices = indices
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.indices]

MAPPING = {
    "0": "@", "1": "A", "2": "B", "3": "C", "4": "D",
    "5": "E", "6": "F", "7": "G", "8": "H", "9": "I"
}

SECRET_KEY = "e2a12f8cfe714c0465322772745326c4"

def deobfuscate_ip(obfuscated):
    if not obfuscated:
        return None
    try:
        decoded = base64.b64decode(obfuscated).decode()
        original_encoded = "".join(
            chr(ord(c) ^ ord(SECRET_KEY[i % len(SECRET_KEY)])) for i, c in enumerate(decoded)
        )
        original_replaced = base64.b64decode(original_encoded).decode()
        inverse_mapping = {v: k for k, v in MAPPING.items()}
        original_ip = "".join(inverse_mapping.get(c, c) for c in original_replaced)
        return original_ip
    except Exception as e:
        print("Errore durante la deoffuscazione:", e)
        return None

# -------------------------------
# Load model and final preprocessor pipeline (includes feature selection)
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    final_preprocessor = joblib.load("preprocessor.pkl")
    return model, final_preprocessor

model, final_preprocessor = load_model()

def get_real_feature_names(final_preprocessor):
    """
    Extract the feature names from the preprocessor step and then select the ones
    used by the feature selector.
    """
    # final_preprocessor is a Pipeline with steps: preprocessor and feature_selector
    original_preprocessor = final_preprocessor.named_steps['preprocessor']
    feature_names = []
    for name, transformer, columns in original_preprocessor.pipeline.transformers_:
        try:
            feature_names.extend(list(transformer.get_feature_names_out(columns)))
        except Exception:
            if isinstance(columns, list):
                feature_names.extend(columns)
            else:
                feature_names.extend([f"{name}_{i}" for i in range(len(columns))])
    feature_selector = final_preprocessor.named_steps['feature_selector']
    indices = feature_selector.indices
    selected_feature_names = [feature_names[i] for i in indices]
    return selected_feature_names

# Display model feature importances using logistic regression coefficients
try:
    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        full_feature_names = get_real_feature_names(final_preprocessor)
        n = min(len(coefficients), len(full_feature_names))
        coef_data = [(full_feature_names[i], coefficients[i]) for i in range(n)]
        coef_data = sorted(coef_data, key=lambda x: abs(x[1]), reverse=True)
        importance_df = pd.DataFrame(coef_data, columns=["Feature", "Coefficient"]).head(5)
    else:
        raise AttributeError("Il modello non possiede l'attributo coef_.")
    
    st.subheader("📊 Top 5 Most Important Features")
    fig = px.bar(
        importance_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title="Top 5 Feature Importances",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Couldn't display feature importance: {e}")

st.title("IP Classification Model")
st.write("Upload a dataset and get prediction probabilities using the trained model.")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load and display the dataset
        data = pd.read_csv(uploaded_file, low_memory=False)
        st.subheader("📄 Uploaded Data")
        st.write(data.head())

        if 'custom_client' not in data.columns:
            st.error("'custom_client' column is missing in the uploaded dataset.")
        else:
            # Deobfuscate custom_client and keep the original values
            data["custom_client_original"] = data["custom_client"].apply(deobfuscate_ip)
            custom_clients = data[["custom_client_original"]].copy()

            # Apply the final preprocessor pipeline (includes feature selection)
            transformed_data = final_preprocessor.transform(data)

            # Predict probabilities and classes
            probabilities = model.predict_proba(transformed_data)
            predicted_classes = model.predict(transformed_data)

            # Build results dataframe
            results_df = custom_clients.copy()
            results_df["prediction_probability"] = probabilities[:, 1]
            results_df["class"] = predicted_classes
            results_df["prediction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.subheader("🔍 Predictions")
            st.write(results_df.head())

            # Function to convert dataframe to CSV for download
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode("utf-8")

            csv = convert_df(results_df)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
