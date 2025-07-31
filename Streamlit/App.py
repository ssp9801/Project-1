import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Cancer Subtype Predictor", layout="wide")
st.title("Cancer Subtype Prediction from Gene Expression")
st.markdown("Upload a CSV for molecular subtype prediction (MDA classification).")

# --- Model & Preprocessor Loading (Cached) ---
@st.cache_resource
def load_model_components():
    models_dir = 'trained_models'
    try:
        model = joblib.load(os.path.join(models_dir, 'random_forest_model.joblib'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.joblib'))
        gene_names = joblib.load(os.path.join(models_dir, 'gene_names.joblib'))
        return model, scaler, label_encoder, gene_names
    except FileNotFoundError as e:
        st.error(f"Error loading model components: {e}. Check 'trained_models' folder.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred loading model: {e}.")
        st.stop()

# Load components
model, scaler, label_encoder, training_gene_names = load_model_components()

# --- File Upload & Validation ---
st.info("""
    **CSV Upload Instructions:**
    - Gene expression data in CSV format.
    - Rows: Samples, Columns: Genes.
    - Gene names (column headers) must match model's training genes.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        N_ROWS = 5000
        # Read only the first N_ROWS from the CSV
        input_df = pd.read_csv(uploaded_file, nrows=N_ROWS)

        st.write(f"### Processing first {N_ROWS} rows...")
        st.write("Preview:")
        st.dataframe(input_df.head())

        # --- Data Preparation ---
        processed_input_data = pd.DataFrame(0.0, index=input_df.index, columns=training_gene_names)
        common_genes = list(set(input_df.columns) & set(training_gene_names))
        processed_input_data[common_genes] = input_df[common_genes]

        st.write("### Scaling Data...")
        scaled_input = scaler.transform(processed_input_data)
        scaled_input_df = pd.DataFrame(scaled_input, columns=training_gene_names, index=input_df.index)
        st.write("Scaling complete.")

        st.write("### Making Predictions...")
        predictions_encoded = model.predict(scaled_input)
        predictions_subtype = label_encoder.inverse_transform(predictions_encoded)

        prediction_df = pd.DataFrame({
            'SampleID': input_df.index,
            'Predicted_Subtype': predictions_subtype
        })
        prediction_df.set_index('SampleID', inplace=True)
        st.write("### Prediction Results:")
        st.dataframe(prediction_df)

    except pd.errors.EmptyDataError:
        st.error("Uploaded CSV file is empty. Please upload data.")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}. Check CSV format.")