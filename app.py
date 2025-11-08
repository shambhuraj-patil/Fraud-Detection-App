import os
import joblib
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

# Load only the trained ML model 
@st.cache_resource
def load_model():
    model_path = "fraud_model.pkl"
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    return model

# Preprocess dataset
@st.cache_data
def preprocess_dataset(dataset):
    dataset["Amount"] = dataset["Amount"].fillna(dataset["Amount"].mean())
    dataset.drop(columns=["TransactionID", "CustomerID"], inplace=True, errors='ignore')

    numeric_columns = ["Amount", "TransactionSpeed"] if "TransactionSpeed" in dataset.columns else ["Amount"]
    for col in numeric_columns:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

    categorical_columns = ["TransactionType", "Location", "DeviceType", "TimeOfDay"]
    existing_cols = [col for col in categorical_columns if col in dataset.columns]
    dataset = pd.get_dummies(dataset, columns=existing_cols, drop_first=True)
    return dataset

# Streamlit Web App
def streamlit_app():
    st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📥 Upload & Preprocess", "📊 Visualization", "🧠 Model Prediction"])

    # ----------------------------- HOME PAGE ----------------------------- #
    if page == "🏠 Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("fraud_banner.jpg", use_container_width=True)

        st.title("🛡️ Credit Card Fraud Detection System")
        st.markdown("Predict the likelihood of fraud in transaction data using a trained Random Forest model.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Features:")
            st.markdown("""
                🔹 Upload your transaction CSV file  
                🔹 Clean and preprocess the data  
                🔹 Visualize fraud trends  
                🔹 Predict potential frauds  
            """)
        with col2:
            st.markdown("### ⚙️ Tech Stack:")
            st.markdown("""
                🔹 Python  
                🔹 Streamlit  
                🔹 Scikit-learn  
                🔹 Pandas, Seaborn  
            """)

        st.markdown("---")
        st.caption("Developed by **Shambhuraj Patil**")

    # ----------------------- UPLOAD & PREPROCESS ------------------------ #
    elif page == "📥 Upload & Preprocess":
        st.title("📥 Upload & Preprocess Data")

        with st.expander("📝 Please read this before uploading your file (Important)", expanded=False):
            st.markdown("""
            - Make sure your CSV file contains the following required columns:  
              `TransactionID`, `CustomerID`, `Amount`, `TransactionType`, `Location`, `DeviceType`, `TimeOfDay`, `Fraud`
            - Missing or renamed columns will prevent the app from working properly.
            """)

        # Offer sample CSV download for user reference
        try:
            sample_dataset = pd.read_csv("fraud_detection.csv").head(20)
            st.download_button("📄 Download Sample Input CSV", sample_dataset.to_csv(index=False),
                            file_name="sample_input.csv", mime="text/csv")
        except FileNotFoundError:
            st.warning("Sample CSV not found. Please check file path or re-upload.")

        st.info("Upload a dataset to clean, remove outliers, and encode categorical variables.")

        upload_file = st.file_uploader("📤 Upload CSV", type="csv")

        if upload_file:
            dataset = pd.read_csv(upload_file)
            st.session_state["raw_data"] = dataset.copy()

            st.success("✅ File Uploaded Successfully")
            st.subheader("📄 Raw Data Preview")
            st.dataframe(dataset.head())

            with st.spinner("⏳ Preprocessing data..."):
                dataset = preprocess_dataset(dataset)
                st.session_state["preprocessed_data"] = dataset

            st.subheader("✅ Preprocessed Data")
            st.dataframe(dataset.head())

    # ---------------------------- VISUALIZATION -------------------------- #
    elif page == "📊 Visualization":
        st.title("📊 Fraud Data Visualization")

        if "raw_data" not in st.session_state:
            st.warning("⚠️ Please upload and preprocess your data first.")
            st.stop()

        dataset = st.session_state["raw_data"]

        tab1, tab2, tab3 = st.tabs(["Fraud vs Non-Fraud", "Fraud by Transaction Type", "Amount Distribution"])

        with tab1:
            st.subheader("Fraud vs Non-Fraud Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(x="Fraud", data=dataset, palette=["green", "red"], ax=ax1)
            st.pyplot(fig1)

        with tab2:
            if "TransactionType" in dataset.columns:
                st.subheader("Fraud by Transaction Type")
                fig2, ax2 = plt.subplots()
                sns.countplot(x="TransactionType", hue="Fraud", data=dataset, palette=["green", "red"], ax=ax2)
                plt.xticks(rotation=45)
                st.pyplot(fig2)

        with tab3:
            if "Amount" in dataset.columns:
                st.subheader("Amount Distribution")
                fig3, ax3 = plt.subplots()
                sns.histplot(data=dataset, x="Amount", hue="Fraud", bins=30, kde=True, ax=ax3)
                st.pyplot(fig3)

    # -------------------------- MODEL PREDICTION -------------------------- #
    elif page == "🧠 Model Prediction":
        st.title("🧠 Model Prediction")

        if "preprocessed_data" not in st.session_state:
            st.warning("⚠️ Please upload and preprocess your data first.")
            st.stop()

        dataset = st.session_state["preprocessed_data"]

        st.info("""
            ⚠️ **Important:**  
            - Dataset must match the model's trained features.  
            - Missing dummy columns will be filled with zeros automatically.
        """)

        model = load_model()
        if model is None:
            st.error("🚫 Model file not found. Please ensure 'fraud_model.pkl' is in the app directory.")
            st.stop()

        # Align dataset columns with model features
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in dataset.columns:
                dataset[col] = 0
        dataset = dataset[model_features]

        # Predict
        predictions = model.predict(dataset)
        probabilities = model.predict_proba(dataset)[:, 1] * 100

        # Combine results
        result_dataset = dataset.copy()
        result_dataset["Fraud Prediction"] = predictions.astype(int)
        result_dataset["Fraud Probability (%)"] = probabilities.round(2)

        st.subheader("🔍 Prediction Results")
        st.dataframe(result_dataset.head(20))

        st.download_button("⬇️ Download Predictions", result_dataset.to_csv(index=False), file_name="fraud_predictions.csv", mime="text/csv")

        st.success(f"✅ Total Transactions Analyzed: {len(result_dataset)}")
        st.info(f"🚨 Fraudulent Transactions Detected: {(result_dataset['Fraud Prediction'] == 1).sum()}")

# Run the App
if __name__ == "__main__":
    streamlit_app()
