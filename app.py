import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
import base64
import numpy as np

from processor import MultiColumnLabelEncoder, ClickDataPreprocessor

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

def get_real_feature_names(preprocessor):
    """Extracts feature names from the ColumnTransformer inside the ClickDataPreprocessor."""
    feature_names = []

    for name, transformer, columns in preprocessor.pipeline.transformers_:
        if isinstance(transformer, OneHotEncoder):
            # OneHotEncoder genera più colonne per feature
            transformed_names = transformer.get_feature_names_out(columns)
        elif isinstance(transformer, TargetEncoder):
            # TargetEncoder restituisce lo stesso numero di colonne
            transformed_names = columns
        elif isinstance(transformer, MultiColumnLabelEncoder):
            # LabelEncoder mantiene i nomi originali
            transformed_names = columns
        elif isinstance(transformer, StandardScaler):
            # StandardScaler applicato a colonne numeriche senza rinomina
            transformed_names = columns
        else:
            # Caso di passthrough
            transformed_names = columns

        feature_names.extend(transformed_names)

    return feature_names

# Se il modello è LogisticRegression, usiamo i coefficienti per visualizzare le "importances"
try:
    # Verifichiamo che il modello abbia l'attributo coef_
    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        # Otteniamo i nomi completi delle feature (come usati nel preprocessor)
        full_feature_names = get_real_feature_names(preprocessor)
        # Se il modello ha salvato anche le feature non nulle, possiamo usarle per filtrare
        # In ogni caso, calcoliamo le importanze come valore assoluto dei coefficienti
        coef_data = [(full_feature_names[i], coefficients[i]) for i in range(len(full_feature_names))]
        # Ordiniamo per valore assoluto decrescente
        coef_data = sorted(coef_data, key=lambda x: abs(x[1]), reverse=True)
        importance_df = pd.DataFrame(coef_data, columns=["Feature", "Coefficient"]).head(5)
    else:
        # Fallback nel caso in cui non si trovi l'attributo coef_
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
        # Carica e visualizza il dataset
        data = pd.read_csv(uploaded_file, low_memory=False)
        st.subheader("📄 Uploaded Data")
        st.write(data.head())

        # Verifica la presenza della colonna 'custom_client'
        if 'custom_client' not in data.columns:
            st.error("'custom_client' column is missing in the uploaded dataset.")
        else:
            # Deoffusca custom_client e salvalo
            data["custom_client_original"] = data["custom_client"].apply(deobfuscate_ip)
            custom_clients = data[["custom_client_original"]].copy()

            # Applica il preprocessor
            transformed_data = preprocessor.transform(data)

            # Predict probabilities e classi
            probabilities = model.predict_proba(transformed_data)
            predicted_classes = model.predict(transformed_data)

            # Costruisci il DataFrame dei risultati
            results_df = custom_clients.copy()
            results_df["prediction_probability"] = probabilities[:, 1]  # probabilità per la classe 1
            results_df["class"] = predicted_classes
            results_df["prediction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Visualizza le predizioni
            st.subheader("🔍 Predictions")
            st.write(results_df.head())

            # Funzione per convertire il DataFrame in CSV per il download
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
