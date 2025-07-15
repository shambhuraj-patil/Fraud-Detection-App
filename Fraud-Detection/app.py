
import pickle
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

def streamlit_app():
    st.set_page_config(page_title="Credit Card Fraud Detection | Shambhuraj", layout="wide")

    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📥 Upload & Preprocess", "📊 Visualization", "🧠 Model Prediction"])

    if page == "🏠 Home":
        # Banner Image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://i.pinimg.com/736x/66/7e/23/667e233091711d3f3e981b9b7c38c72b.jpg", use_container_width=True)

        st.title("🛡️ Credit Card Fraud Detection System")
        st.markdown("Predict the likelihood of fraud in transaction data using a trained machine learning model.")

        col1,col2 = st.columns(2)
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
            st.markdown("###  🧭 How to Use:")
            st.markdown("""
                🔹 Upload a CSV file  
                🔹 Go to 'Upload & Preprocess Data'  
                🔹 Explore 'Visualization'  
                🔹 Use 'Model Prediction'  
                🔹 Download your fraud results  
            """)

        st.markdown("---")
        st.markdown("Made with ❤️ using Python, Streamlit, Scikit-learn  \nBy **Shambhuraj Patil**")
    
    elif page == "📥 Upload & Preprocess":
        st.title("📥 Upload & Preprocess Data")

        with st.expander("📝 **Please read this before uploading your file (Important)**", expanded=False):
            st.markdown("""
            - Download the sample CSV first (sample_input.csv)  
            - Your file must follow the same column structure
            - The app won't work if columns are missing or renamed
            """)
        # Load and offer first 10 rows from training data as sample
        try:
            sample_dataset = pd.read_csv("Fraud-Detection/fraud_detection.csv").head(20)
            st.download_button("📄 Download Sample Input CSV", sample_dataset.to_csv(index=False),
                            file_name="sample_input.csv", mime="text/csv")
        except FileNotFoundError:
            st.warning("Sample CSV not found. Please check file path or re-upload.")

        upload_file = st.file_uploader("📤 Upload a CSV file", type="csv")
        if upload_file:
            try:
                dataset = pd.read_csv(upload_file)
                st.success("✅ File Uploaded Successfully")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
            
            # Strict column check 
            required_columns = ["TransactionID", "CustomerID", "Amount", "TransactionType",
                                "Location", "DeviceType", "TimeOfDay", "Fraud"]

            missing_cols = [col for col in required_columns if col not in dataset.columns]
            
            if missing_cols:
                st.error(f"❌ Your dataset is missing required columns: {missing_cols}")
                st.markdown("Download the correct sample format above and match your columns.")
                st.stop()

            st.subheader("📄 Raw Data Preview")
            st.dataframe(dataset.head(10))
            st.session_state["raw_data"] = dataset.copy()

            # Fill missing 'Amount' values
            dataset["Amount"] = dataset["Amount"].fillna(dataset["Amount"].mean())
            st.info("ℹ️ Filled missing 'Amount' values with mean")

            # Drop unneeded columns
            dataset.drop(columns=["TransactionID", "CustomerID"], inplace=True)
            st.info("🧹 Dropped 'TransactionID' and 'CustomerID'")

            # Removing outliers using the IQR (Interquartile Range) method
            numeric_columns = ["Amount","TransactionSpeed"]
            for col in numeric_columns:
                Q1 = dataset[col].quantile(0.25)
                Q3 = dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]
            st.info("📉 Removed Outliers")

            # Encode categorical columns
            categorical_columns = ["TransactionType", "Location", "DeviceType", "TimeOfDay"]
            dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)
            st.info("🔠 One-hot encoded categorical columns")

            # Convert bool to int
            bool_cols = dataset.select_dtypes(include='bool').columns
            dataset[bool_cols] = dataset[bool_cols].astype(int)

            st.session_state["preprocessed_data"] = dataset
            st.subheader("✅ Preprocessed Data")
            st.dataframe(dataset)

    elif page == "📊 Visualization":
        st.title("📊 Fraud Data Visualization")

        if "raw_data" not in st.session_state:
            st.warning("⚠️ Please upload and preprocess your data first.")
            st.stop()
        else:
            dataset = st.session_state["raw_data"]

        tab1,tab2 = st.tabs(["Fraud vs Non-Fraud","Fraud by Transaction Type"])
        with tab1:
            st.subheader("📉 Fraud vs Non-Fraud Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(x="Fraud",hue="Fraud", data=dataset, palette=["green", "red"], ax=ax1)
            st.pyplot(fig1)

        with tab2:
            st.subheader("💳 Fraud Distribution by Transaction Type")
            fig2, ax2 = plt.subplots()
            sns.countplot(x="TransactionType", hue="Fraud", data=dataset, palette=["green", "red"], ax=ax2)
            plt.xticks(rotation=45)
            st.pyplot(fig2)

    elif page == "🧠 Model Prediction":
        st.title("🧠 Model Prediction")

        if "preprocessed_data" not in st.session_state:
            st.warning("⚠️ Please upload and preprocess your data first.")
            st.stop()

        dataset = st.session_state["preprocessed_data"]

        # Load model and scaler
        try:
            model = pickle.load(open("Fraud-Detection/fraud_model.pkl", "rb"))
            scaler = pickle.load(open("Fraud-Detection/scaler.pkl", "rb"))
        except FileNotFoundError:
            st.error("🚫 fraud_model.pkl' or 'scaler.pkl' not found.")
            st.stop()

        # Align dataset with model input
        model_features = model.feature_names_in_
        data = dataset.copy()
        for col in model_features:
            if col not in data.columns:
                data[col] = 0  # Add missing columns as 0

        data = data[model_features]

        # Scale and predict
        scaled_data = scaler.transform(data)
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)[:, 1] * 100

        # Add predictions to copy of dataset
        result_dataset = dataset.copy()
        result_dataset["Fraud Prediction"] = predictions
        result_dataset["Fraud Probability (%)"] = probabilities.round(2)

        st.subheader("🔍 Prediction Results")
        st.dataframe(result_dataset.head(20))

        st.markdown("---")
        st.markdown("📥 **Download the predicted results here:**")

        download = st.download_button("Download Predictions", result_dataset.to_csv(index=False),
                           file_name="fraud_predictions.csv", mime="text/csv",)

        if download:
            st.success("✅ Predictions file downloaded successfully!")
        
        st.success(f"✅ Total transactions analyzed: {len(result_dataset)}")
        st.info(f"🚨 Fraudulent transactions detected: {(result_dataset['Fraud Prediction'] == 1).sum()}")

        st.session_state["predicted_data"] = result_dataset
        
# Run the app
streamlit_app()
