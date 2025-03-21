import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

import base64

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

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

st.title("IP Classification Model")
st.write("Upload a dataset and get prediction probabilities using the trained model.")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load and display dataset
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.write(data.head())

        # Backup 'custom_client' for final output
        if 'custom_client' not in data.columns:
            st.error("'custom_client' column is missing in the uploaded dataset.")
        else:
            # Deoffusca custom_client e salvalo
            data["custom_client_original"] = data["custom_client"].apply(deobfuscate_ip)
            custom_clients = data[["custom_client_original"]].copy()

            # Apply preprocessing
            transformed_data = preprocessor.transform(data)

            # Predict probabilities
            probabilities = model.predict_proba(transformed_data)
            predicted_classes = model.predict(transformed_data)

            # Build result DataFrame
            results_df = custom_clients.copy()
            results_df["prediction_probability"] = probabilities[:, 1]  # probability for class 1
            results_df["class"] = predicted_classes
            results_df["prediction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Show predictions
            st.subheader("🔍 Predictions")
            st.write(results_df.head())

            # Download results
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
