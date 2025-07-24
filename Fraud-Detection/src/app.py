import os
import pickle
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

# Load the trained ML model and the scaler object
# Returns the model and scaler if they exist, otherwise returns None
# Caching model and scaler for performance

@st.cache_resource
def load_model_and_scaler():
    model_path = "models/fraud_model.pkl"
    scaler_path = "models/scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# Preprocess the dataset: fill missing values, remove outliers, encode categorical columns
# Caching preprocessing of uploaded dataset

@st.cache_data
def preprocess_dataset(dataset):
    # Fill missing values in 'Amount' column with the mean
    dataset["Amount"] = dataset["Amount"].fillna(dataset["Amount"].mean())

    # Drop ID-related columns if they exist (not useful for prediction)
    dataset.drop(columns=["TransactionID", "CustomerID"], inplace=True, errors='ignore')

    # Remove outliers from numeric columns using the IQR method
    numeric_columns = ["Amount", "TransactionSpeed"] if "TransactionSpeed" in dataset.columns else ["Amount"]
    for col in numeric_columns:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

    # Convert categorical variables into dummy/one-hot encoded columns
    categorical_columns = ["TransactionType", "Location", "DeviceType", "TimeOfDay"]
    existing_cols = [col for col in categorical_columns if col in dataset.columns]
    dataset = pd.get_dummies(dataset, columns=existing_cols, drop_first=True)

    return dataset

# Main Streamlit web application

def streamlit_app():
    st.set_page_config(page_title="Credit Card Fraud Detection | Shambhuraj", layout="wide")
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📥 Upload & Preprocess", "📊 Visualization", "🧠 Model Prediction"])

    # ----------------------------- HOME PAGE ----------------------------- #
    if page == "🏠 Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("assets/fraud_banner.jpg", use_container_width=True)

        st.title("🛡️ Credit Card Fraud Detection System")
        st.markdown("Predict the likelihood of fraud in transaction data using a trained machine learning model.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ What You Can Do:")
            st.markdown("""
                🔹 Upload your transaction CSV file  
                🔹 Preprocess and clean the data  
                🔹 Visualize fraud patterns  
                🔹 Predict fraud using ML  
                🔹 Download results with probabilities  
            """)

        with col2:
            st.markdown("### 🧭 How to Use:")
            st.markdown("""
                🔹 Upload a CSV file  
                🔹 Go to 'Upload & Preprocess Data'  
                🔹 Explore 'Visualization'  
                🔹 Use 'Model Prediction'  
                🔹 Download your fraud results  
            """)

        st.markdown("---")
        st.markdown("Made with ❤️ using Python, Streamlit, Scikit-learn  \nBy **Shambhuraj Patil**")

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
            sample_dataset = pd.read_csv("data/fraud_detection.csv").head(20)
            st.download_button("📄 Download Sample Input CSV", sample_dataset.to_csv(index=False),
                            file_name="sample_input.csv", mime="text/csv")
        except FileNotFoundError:
            st.warning("Sample CSV not found. Please check file path or re-upload.")

        upload_file = st.file_uploader("📤 Upload a CSV file", type="csv")

        if upload_file:
            dataset = pd.read_csv(upload_file)
            st.session_state["raw_data"] = dataset.copy()  # Save raw data for visualization

            # Validate column presence
            required_columns = ["TransactionID", "CustomerID", "Amount", "TransactionType",
                                "Location", "DeviceType", "TimeOfDay", "Fraud"]

            missing_cols = [col for col in required_columns if col not in dataset.columns]
            if missing_cols:
                st.error(f"❌ Your dataset is missing required columns: {missing_cols}")
                st.markdown("📌 Please ensure your file has all required columns. You can download a sample format above.")
                st.stop()

            st.success("✅ File Uploaded Successfully")
            st.subheader("📄 Raw Data Preview")
            st.dataframe(dataset.head(10))

            # Apply preprocessing steps with spinner
            with st.spinner("⏳ Processing your data..."):
                dataset = preprocess_dataset(dataset)
                st.session_state["preprocessed_data"] = dataset

            st.subheader("✅ Preprocessed Data")
            st.dataframe(dataset.head(10))

    # ---------------------------- VISUALIZATION -------------------------- #
    elif page == "📊 Visualization":
        st.title("📊 Fraud Data Visualization")

        if "preprocessed_data" not in st.session_state:
            st.warning("⚠️ Please upload and preprocess your data first.")
            st.stop()

        dataset = st.session_state["raw_data"]  # Use raw data to retain original categorical columns

        tab1, tab2, tab3, = st.tabs(["Fraud vs Non-Fraud", "Fraud by Transaction Type", "Amount"])

        with tab1:
            st.subheader("📉 Fraud vs Non-Fraud Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(x="Fraud", hue="Fraud", data=dataset, palette=["green", "red"], ax=ax1)
            st.pyplot(fig1)

        with tab2:
            if "TransactionType" in dataset.columns:
                st.subheader("💳 Fraud Distribution by Transaction Type")
                fig2, ax2 = plt.subplots()
                sns.countplot(x="TransactionType", hue="Fraud", data=dataset, palette=["green", "red"], ax=ax2)
                plt.xticks(rotation=45)
                st.pyplot(fig2)
            else:
                st.info("'TransactionType' column not found in your dataset.")

        with tab3:
            if "Amount" in dataset.columns:
                st.subheader("💰 Amount Distribution")
                fig3, ax3 = plt.subplots()
                sns.histplot(data=dataset, x="Amount", hue="Fraud", bins=40, kde=True, ax=ax3)
                st.pyplot(fig3)

    # -------------------------- MODEL PREDICTION -------------------------- #
    elif page == "🧠 Model Prediction":
        st.title("🧠 Model Prediction")

        if "preprocessed_data" not in st.session_state:
            st.warning("⚠️ Please upload and preprocess your data first.")
            st.stop()

        dataset = st.session_state["preprocessed_data"]

        st.markdown("### 🧾 Input Format Notice")
        st.info(
            """⚠️ **Important:**  
            - Your preprocessed dataset must match the model's trained feature structure.  
            - Unknown categories or missing dummy columns will be automatically filled with zeros.  
            - Make sure preprocessing steps like encoding and outlier removal were applied."""
        )

        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            st.error("🚫 Model or Scaler files not found. Make sure 'fraud_model.pkl' and 'scaler.pkl' are in the working directory.")
            st.stop()

        # Align input dataset with model features
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in dataset.columns:
                dataset[col] = 0  # Add missing columns as 0
        dataset = dataset[model_features]  # Reorder columns

        # Apply scaling and prediction
        scaled_data = scaler.transform(dataset)
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)[:, 1] * 100

        # Attach results to the dataset
        result_dataset = dataset.copy()
        result_dataset["Fraud Prediction"] = predictions.astype(int)
        result_dataset["Fraud Probability (%)"] = probabilities.round(2)

        st.subheader("🔍 Prediction Results")

        # Add checkbox to control how many rows to show
        if st.checkbox("Show all prediction results", value=False):
            st.dataframe(result_dataset[["Fraud Prediction", "Fraud Probability (%)"]])
        else:
            st.dataframe(result_dataset[["Fraud Prediction", "Fraud Probability (%)"]].head(20))
        
        st.markdown("---")
        st.download_button("Download Predictions", result_dataset.to_csv(index=False), file_name="fraud_predictions.csv", mime="text/csv")
        st.success(f"✅ Total transactions analyzed: {len(result_dataset)}")
        st.info(f"🚨 Fraudulent transactions detected: {(result_dataset['Fraud Prediction'] == 1).sum()}")

# Run the app
if __name__ == "__main__":
    streamlit_app()
